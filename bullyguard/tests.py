from bullyguard.data_modules.transformations import HuggingFaceTokenizationTransformation
from bullyguard.models import backbones

pretrained_tokenizer_name_or_path = "gs://bullyguard/data/processed/rebalanced_splits/trained_tokenizer"
max_sequence_length = 72
tokenizer = HuggingFaceTokenizationTransformation(pretrained_tokenizer_name_or_path, max_sequence_length)

texts = ["hi, how are you?"]

encoding = tokenizer(texts)

backbone = backbones.HuggingFaceBackbone(pretrained_model_name_or_path="bert-base-uncased", pretrained=False)

output = backbone(encoding).pooler_output

print(output)
print(output.shape)
