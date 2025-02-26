from bullyguard.data_modules.transformations import HuggingFaceTokenizationTransformation

pretrained_tokenizer_name_or_path = "gs://bullyguard/data/processed/rebalanced_splits/trained_tokenizer"
max_sequence_length = 72
tokenizer = HuggingFaceTokenizationTransformation(pretrained_tokenizer_name_or_path, max_sequence_length)

texts = ["hi, how are you?"]

output = tokenizer(texts)

print(f"{output=}")
