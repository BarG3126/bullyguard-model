# from bullyguard.data_modules.transformations import HuggingFaceTokenizationTransformation
# from bullyguard.models import backbones

# pretrained_tokenizer_name_or_path = "gs://bullyguard/data/processed/rebalanced_splits/trained_tokenizer"
# max_sequence_length = 72
# tokenizer = HuggingFaceTokenizationTransformation(pretrained_tokenizer_name_or_path, max_sequence_length)

# texts = ["hi, how are you?"]

# encoding = tokenizer(texts)

# backbone = backbones.HuggingFaceBackbone(pretrained_model_name_or_path="bert-base-uncased", pretrained=False)

# output = backbone(encoding).pooler_output

# print(output)
# print(output.shape)

import hydra

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bullyguard.config_schemas.training.training_task_schemas import setup_config

setup_config()


@hydra.main(config_name="test_training_task", version_base=None)
def main(config: DictConfig) -> None:
    print(60 * "#")
    print(OmegaConf.to_yaml(config))
    print(60 * "#")

    # model = instantiate(config)
    # print(model)

    # texts = ["hello, how are you?"]
    # encodings = model.backbone.transformation(texts)

    # output = model(encodings)
    # print(f"{output.shape=}")


if __name__ == "__main__":
    main()
