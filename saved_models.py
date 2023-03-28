from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

config = RobertaConfig.from_pretrained("microsoft/codebert-base")

model = RobertaModel.from_pretrained("microsoft/codebert-base", config, ignore_mismatched_sizes=True)

model.save_pretrained("./saved_models/model_bert")

config = model.config
config.save_pretrained("./saved_models/model.config")

tokenizer= RobertaTokenizer.from_pretrained("microsoft/codebert-base")
tokenizer.save_pretrained("./saved_models/tokenizer_bert")

