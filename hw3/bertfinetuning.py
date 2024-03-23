
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from datasets import load_dataset
from accelerate import Accelerator
import evaluate
from tqdm.auto import tqdm
import numpy as np
import collections
import json
import os

data_dir = './Spoken-SQuAD/'
train_data_name = 'spoken_train-v1.1.json'
test_data_name = 'spoken_test-v1.1.json' 
test_wer44_data_name = 'spoken_test-v1.1_WER44.json' 
test_wer54_data_name = 'spoken_test-v1.1_WER54.json' 

def refine_json(file_name, data_dir, output_prefix='refined_'):
    input_file_path = os.path.join(data_dir, file_name)
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    examples = []
    for elem in data['data']:
        title = elem['title'].strip()

        for paragraph in elem['paragraphs']:
            context = paragraph['context'].strip()

            for qa in paragraph['qas']:
                example = {
                    'id': qa['id'],
                    'title': title,
                    'context': context,
                    'question': qa['question'].strip(),
                    'answers': {
                        'answer_start': [answer["answer_start"] for answer in qa['answers']],
                        'text': [answer["text"].strip() for answer in qa['answers']]
                    }
                }
                examples.append(example)
    
    out_dict = {'data': examples}
    output_file_path = os.path.join(data_dir, output_prefix + file_name)
    
    with open(output_file_path, 'w') as f:
        json.dump(out_dict, f, indent=4)

    return output_file_path

def preprocess_training_examples(examples, tokenizer, max_sentence_length, stride):
    questions = [question.strip() for question in examples['question']]
    tokens = tokenizer(
        questions, 
        examples['context'],
        max_length = max_sentence_length,
        truncation = 'only_second',
        stride = stride, 
        return_overflowing_tokens = True,
        return_offsets_mapping=True, 
        padding = 'max_length'
    )

    offset_mapping = tokens.pop('offset_mapping')
    sample_map = tokens.pop('overflow_to_sample_mapping')
    answers = examples['answers']
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer["text"][0])
        sequence_ids = tokens.sequence_ids(i)

        # find start and end of the context
        idx = 0
        while sequence_ids[idx] != 1: 
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    tokens['start_positions'] = start_positions
    tokens['end_positions'] = end_positions
    return tokens

def process_validation_examples(examples, tokenizer, max_sentence_length, stride):
    questions = [question.strip() for question in examples['question']]
    tokens = tokenizer(
        questions, 
        examples['context'],
        max_length = max_sentence_length,
        truncation = 'only_second',
        stride = stride, 
        return_overflowing_tokens = True,
        return_offsets_mapping=True, 
        padding = 'max_length'
    )

    sample_map = tokens.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(tokens['input_ids'])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = tokens.sequence_ids(i)
        offsets = tokens['offset_mapping'][i]
        tokens["offset_mapping"][i] = [
            offset if sequence_ids[k] == 1 else None for k, offset in enumerate(offsets)
        ]

    tokens['example_id'] = example_ids
    return tokens

def compute_metrics(start_logits, end_logits, features, examples, metric, n_best = 20, max_answer_length = 30):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features): 
        example_to_features[feature["example_id"]].append(idx)
    
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        
        for feature_index in example_to_features[example_id]: 
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            
            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes: 
                for end_index in end_indexes: 
                    if offsets[start_index] is None or offsets[end_index] is None: 
                        continue

                    if end_index < start_index or end_index-start_index+1 > max_answer_length: 
                        continue
                    
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index]
                    }
                    answers.append(answer)
        # select answer with best score
        if len(answers) > 0: 
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else: 
            predicted_answers.append({"id": example_id, "prediction_text": ""})
        
    gt_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=gt_answers)

def evaluate_model(model, dataloader, dataset, dataset_before_preprocessing, accelerator=None, metric=None):
    if not accelerator: 
        accelerator = Accelerator(mixed_precision='fp16')
        model, dataloader = accelerator.prepare(
            model, dataloader
        )
    
    model.eval()
    start_logits = []
    end_logits = []
    for batch in tqdm(dataloader):
        with torch.no_grad(): 
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(dataset)]
    end_logits = end_logits[: len(dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, dataset, dataset_before_preprocessing, metric
    )
    return metrics

def train_model(model, train_dataloader, eval_dataloader, epochs = 1, gradient_accumulation_steps=1, output_dir='./bert-base-uncased-finetuned-spoken-squad'):
    training_steps = epochs * len(train_dataloader)

    accelerator = Accelerator(mixed_precision='fp16')
    optimizer = AdamW(model.parameters(), lr = 2e-5)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=training_steps,
    )

    progress_bar = tqdm(range(training_steps))
    metric = evaluate.load("squad")

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            # print(step)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)

        # evaluate after each epoch 
        results = evaluate_model(model, eval_dataloader, validation_dataset, spoken_squad_dataset['validation'], accelerator, metric)
        print(f"epoch {epoch}:", results)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

if __name__ == "__main__":
    train_data = refine_json(train_data_name, data_dir)
    test_data = refine_json(test_data_name, data_dir)
    test_wer44_data = refine_json(test_wer44_data_name, data_dir)
    test_wer54_data = refine_json(test_wer54_data_name, data_dir)

    spoken_squad_dataset = load_dataset('json',
                                        data_files= { 'train': train_data,
                                                    'validation': test_data,         
                                                    'test_WER44': test_wer44_data,   
                                                    'test_WER54': test_wer54_data },
                                        field = 'data')

    model_checkpoint = "bert-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    max_sentence_length = 384 
    stride = 64

    print("Preprocessing training data...")
    train_dataset = spoken_squad_dataset['train'].map(
        lambda examples: preprocess_training_examples(examples, tokenizer, max_sentence_length, stride),
        batched = True,
        remove_columns=spoken_squad_dataset['train'].column_names
    )

    validation_dataset = spoken_squad_dataset['validation'].map(
        lambda examples: process_validation_examples(examples, tokenizer, max_sentence_length, stride),
        batched = True,
        remove_columns=spoken_squad_dataset['validation'].column_names
    )

    test_WER44_dataset = spoken_squad_dataset['test_WER44'].map(
        lambda examples: process_validation_examples(examples, tokenizer, max_sentence_length, stride),
        batched = True,
        remove_columns=spoken_squad_dataset['test_WER44'].column_names
    )

    test_WER54_dataset = spoken_squad_dataset['test_WER54'].map(
        lambda examples: process_validation_examples(examples, tokenizer, max_sentence_length, stride),
        batched = True,
        remove_columns=spoken_squad_dataset['test_WER54'].column_names
    )


    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    test_WER44_set = test_WER44_dataset.remove_columns(["example_id", "offset_mapping"])
    test_WER54_set = test_WER54_dataset.remove_columns(["example_id", "offset_mapping"])

    train_dataset.set_format("torch")   
    validation_set.set_format("torch")
    test_WER44_set.set_format("torch")
    test_WER54_set.set_format("torch")

    train_dataloader = DataLoader(train_dataset, shuffle = True, collate_fn=default_data_collator, batch_size=8)
    eval_dataloader = DataLoader(validation_set, collate_fn=default_data_collator, batch_size=8)
    
    train_model(model, train_dataloader, eval_dataloader, epochs=2, gradient_accumulation_steps=2)

    metric = evaluate.load("squad")
    test_WER44_dataloader = DataLoader(test_WER44_set, collate_fn=default_data_collator, batch_size=8)
    test_WER54_dataloader = DataLoader(test_WER54_set, collate_fn=default_data_collator, batch_size=8)
    test_metrics = evaluate_model(model, eval_dataloader, validation_dataset, spoken_squad_dataset['validation'], metric = metric)
    test_v1_metrics = evaluate_model(model, test_WER44_dataloader, test_WER44_dataset, spoken_squad_dataset['test_WER44'], metric = metric)
    test_v2_metrics = evaluate_model(model, test_WER54_dataloader, test_WER54_dataset, spoken_squad_dataset['test_WER54'], metric = metric)

    print("The Result of Test Set " + str(test_metrics['exact_match']) + ", F1 score: " + str(test_metrics['f1']))
    print("The Result of Test V1 Set " + str(test_v1_metrics['exact_match']) + ", F1 score: " + str(test_v1_metrics['f1']))
    print("The Result of Test V2 Set " + str(test_v2_metrics['exact_match']) + ", F1 score: " + str(test_v2_metrics['f1']))