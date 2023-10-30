import random
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, IdeficsForVisionText2Text, AutoProcessor
from torch.utils.data import IterableDataset
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)


class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control





def create_datasets(tokenizer, args, processor):
    def convert_to_rgb(image):
        if image.mode == "RGB":
            return image
    
        image_rgba = image.convert("RGBA")
        background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, image_rgba)
        alpha_composite = alpha_composite.convert("RGB")
        return alpha_composite

    def ds_transforms(example_batch):
        root_path = args.image_dataset_name
        image_size = processor.image_processor.image_size
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std
    
        image_transform = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
    
        batch_prompt = []
    
        for images in example_batch['choices']:
            images = images.split()
            prompt = []
            for i in range(len(images) // 2):
                base_image = root_path +  '/' + images[2 * i]
                personalized_image = root_path +  '/' + images[2 * i + 1]
                ascii = ord(base_image[-5])
                #if not os.path.isfile(base_image) or not os.path.isfile(personalized_image):
                #    with open('/datasets/personalized_data_sd/final/hell.txt', 'a') as f:
                #        f.write('ga\n')
                #    continue
                if ascii % 2 == 0:
                    prompt.append(base_image)
                    prompt.append("User: What is user's score for this image?")
                    prompt.append("<end_of_utterance>")
                    prompt.append("\nAssistant: User's score for this image: -")
    
                    prompt.append(personalized_image)
                    prompt.append("User: What is user's score for this image?")
                    prompt.append("<end_of_utterance>")
                    prompt.append("\nAssistant: User's score for this image: +")
                else:
                    prompt.append(personalized_image)
                    prompt.append("User: What is user's score for this image?")
                    prompt.append("<end_of_utterance>")
                    prompt.append("\nAssistant: User's score for this image: +")
    
                    prompt.append(base_image)
                    prompt.append("User: What is user's score for this image?")
                    prompt.append("<end_of_utterance>")
                    prompt.append("\nAssistant: User's score for this image: -")
    
            batch_prompt.append(prompt)
    
        inputs = processor(batch_prompt, transform=image_transform, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"]
    
        return inputs
    
    dataframe = pd.read_csv(args.dataset_name)
    user_images = defaultdict(dict)
    for i, row in dataframe.iterrows():
        user_images[i] = ''
        for j in range(51, 51+21 + 4):
            if isinstance(row[dataframe.keys()[j]], float):
                continue
            user_images[i] = user_images[i] + ' ' + row[dataframe.keys()[j]]

    user_choices_df = pd.DataFrame.from_dict(user_images, orient='index', columns=['choices'])
    train_ds = Dataset.from_pandas(user_choices_df)
    ds = train_ds.train_test_split(test_size=0.1)
    train_ds = ds['train']
    test_ds = ds['test']

    train_ds.remove_columns('__index_level_0__')
    test_ds.remove_columns('__index_level_0__')
    train_ds.set_transform(ds_transforms)
    test_ds.set_transform(ds_transforms)
    print(f"Size of the train set: {len(train_ds)}. Size of the validation set: {len(test_ds)}")

    return train_ds, test_ds


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = "auto"  # {"": 0}

    checkpoint = args.model_name
    model = IdeficsForVisionText2Text.from_pretrained(
        checkpoint,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        if (args.use_4bit_qunatization or args.use_8bit_qunatization) and args.use_peft_lora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)

        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(checkpoint)

    return model, peft_config, processor


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
