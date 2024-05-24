import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from datasets import load_metric
import numpy as np
import torch
from tqdm import tqdm

#Se importan funciones auxiliares tomadas de: https://www.kaggle.com/code/noxusdarius/notebook614d02005b

def postprocess_text(preds: list, labels: list) -> tuple:
    """Performs post processing on the prediction text and labels"""

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def prep_data_for_model_fine_tuning(source_lang: list, target_lang: list) -> list:
    """Takes the input data lists and converts into translation list of dicts"""

    data_dict = dict()
    data_dict[TRANSLATION] = []

    for sr_text, tr_text in zip(source_lang, target_lang):
        temp_dict = dict()
        temp_dict[WAYU] = sr_text
        temp_dict[SPANISH] = tr_text

        data_dict[TRANSLATION].append(temp_dict)
    return data_dict



def generate_model_ready_dataset(dataset: list, source: str, target: str,
                                 model_checkpoint: str,
                                 tokenizer: AutoTokenizer):
    """Makes the data training ready for the model"""

    preped_data = []

    for row in dataset:
        inputs = PREFIX + row[source]
        targets = row[target]

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)

        model_inputs[TRANSLATION] = row

        # setup the tokenizer for targets
        labels = tokenizer(targets, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)
        model_inputs[LABELS] = labels[INPUT_IDS]

        preped_data.append(model_inputs)

    return preped_data

def compute_metrics(eval_preds: tuple) -> dict:
    """computes bleu score and other performance metrics """

    metric = load_metric("sacrebleu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {BLEU: result[SCORE]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    result[GEN_LEN] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result
def prep_data_for_model_fine_tuning(source_lang: list, target_lang: list) -> list:
    """Takes the input data lists and converts into translation list of dicts"""

    data_dict = dict()
    data_dict[TRANSLATION] = []

    for sr_text, tr_text in zip(source_lang, target_lang):
        temp_dict = dict()
        temp_dict[WAYU] = sr_text
        temp_dict[SPANISH] = tr_text

        data_dict[TRANSLATION].append(temp_dict)
    return data_dict



def generate_model_ready_dataset(dataset: list, source: str, target: str,
                                 model_checkpoint: str,
                                 tokenizer: AutoTokenizer):
    """Makes the data training ready for the model"""

    preped_data = []

    for row in dataset:
        inputs = PREFIX + row[source]
        targets = row[target]

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)

        model_inputs[TRANSLATION] = row

        # setup the tokenizer for targets
        labels = tokenizer(targets, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)
        model_inputs[LABELS] = labels[INPUT_IDS]

        preped_data.append(model_inputs)

    return preped_data

#Implementación para importar datos

def import_dataset():
    # Se importan los datos en .csv
    info_spanish = pd.read_csv("Datos_espannol.csv", sep=';', encoding='latin-1')
    info_guac = pd.read_csv("Datos_guac.csv", sep=';')

    print("Se han importado los datos")
    # Se crean los conjuntos X y Y para predicciones
    X = info_guac['Oracion']
    y = info_spanish['Oracion']
    # Se separan en train, test y val
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True, random_state=100)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True, random_state=100)

    print("Se han creado los conjuntos de test, train y validacion ")

    training_data = prep_data_for_model_fine_tuning(x_train.values, y_train.values)
    validation_data = prep_data_for_model_fine_tuning(x_val.values, y_val.values)
    test_data = prep_data_for_model_fine_tuning(x_test.values, y_test.values)

    print("Se ha preparado la base de datos para fine-tuning")


    train_data = generate_model_ready_dataset(dataset=training_data[TRANSLATION],
                                              tokenizer=tokenizer,
                                              source=WAYU,
                                              target=SPANISH,

                                              model_checkpoint=MODEL_CHECKPOINT)

    validation_data = generate_model_ready_dataset(dataset=validation_data[TRANSLATION],
                                                   tokenizer=tokenizer,
                                                   source=WAYU,
                                                   target=SPANISH,
                                                   model_checkpoint=MODEL_CHECKPOINT)

    test_data = generate_model_ready_dataset(dataset=test_data[TRANSLATION],
                                             tokenizer=tokenizer,
                                             source=WAYU,
                                             target=SPANISH,
                                             model_checkpoint=MODEL_CHECKPOINT)

    train_df = pd.DataFrame.from_records(train_data)
    validation_df = pd.DataFrame.from_records(validation_data)
    test_df = pd.DataFrame.from_records(test_data)

    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, validation_dataset, test_dataset

def set_trainer(train_dataset, validation_dataset, test_dataset, EPOCH, LR, BATCH_SIZE):

    model_args = Seq2SeqTrainingArguments(
        f"{MODEL_NAME}-finetuned-{SOURCE_LANG}-to-{TARGET_LANG}",
        evaluation_strategy=EPOCH,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.02,
        save_total_limit=3,
        num_train_epochs=12,
        predict_with_generate=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        model_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer, model

def train_model(trainer, model, test_dataset, name):
    print("Train model named: ", name)
    trainer.train()

    trainer.save_model(name)
    test_results = trainer.predict(test_dataset)
    print("Test Bleu Score: ", test_results.metrics["test_bleu"])

    #Se guardan los resultados
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    test_input = test_dataset[TRANSLATION]
    print(test_input)

    for input_text in tqdm(test_input):
        source_sentence = input_text[WAYU]
        encoded_source = tokenizer(source_sentence,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True)
        encoded_source.to(device)  # Move input tensor to the same device as the model

        translated = model.generate(input_ids=encoded_source['input_ids'],
                                    attention_mask=encoded_source['attention_mask'],
                                    max_length=128,  # Adjust as needed
                                    num_beams=4,  # Adjust as needed
                                    early_stopping=True)  # Adjust as needed

        predictions.append([tokenizer.decode(t, skip_special_tokens=True) for t in translated][0])

    #Se genera un dataframe con las predicciones
    ground = []
    input_text = []

    for each_data in tqdm(test_dataset["translation"]):
        ground.append(each_data[SPANISH])
        input_text.append(each_data[WAYU])

    df = pd.DataFrame({"Prediccion:": predictions, "Input in wayu": input_text, "Real spanish sentence: ": ground})
    df.to_excel(name + ".xlsx", index=False)

if __name__ == "__main__":
    #Se inicializan constantes
    TRANSLATION = "translation"
    WAYU = "guac"
    SPANISH = "sp"
    PREFIX = ""
    MAX_INPUT_LENGTH = 128
    MAX_TARGET_LENGTH = 128
    LABELS = "labels"
    INPUT_IDS = "input_ids"
    MODEL_CHECKPOINT = "google-t5/t5-small"

    BATCH_SIZE = 5
    BLEU = "bleu"
    WAYU_TEXT = "guact"
    EPOCH = "epoch"
    GEN_LEN = "gen_len"
    MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]

    SPANISH_TEXT = "spa"
    SCORE = "score"
    SOURCE_LANG = "guac"
    TARGET_LANG = "sp"

    #Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    #Generate dataset
    train_dataset, validation_dataset, test_dataset = import_dataset()

    for EPOCH in [20, 15, 10]:
        for LR in [1e-5, 2e-5, 4e-5]:
            for BATCH_SIZE in [25, 6, 10]:
                name = f"model_epoch{EPOCH}_LR{LR}_BATCH{BATCH_SIZE}"
                trainer, model = set_trainer(train_dataset, validation_dataset, test_dataset, EPOCH, LR, BATCH_SIZE)
                train_model(trainer, model, test_dataset, name)



