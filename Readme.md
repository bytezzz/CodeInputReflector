```
.
├── InputReflector.ipynb  # Code For InputReflector.
├── Siamese.py            # Code For Siamse Network and Quadruplet Network
├── adv_examples.jsonl    # part of Adv_examples to train Sia and Quad Network
├── model.py              # CodeBert Model
├── process_ga_csv.py     # Convert Alert's CSV file to adv_examples.jsonl
├── reveal_non_vul.jsonl  # Reveal Dataset, Non-vulnerable part
├── reveal_vul.jsonl      # Reveal Dataset, vulnerable part
├── test.jsonl            # Devign Dataset, testset
├── train.jsonl           # Devign Dataset, trainset
├── train.py              # Script to train SiamseNetwork and Quadruplet Network
├── triplet_loss.py       # Triplet loss used in training
├── utils.py              # Utils
└── valid.jsonl           # Devign Dataset, validset
```

---

Download Fine-tuned CodeBert Model First:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=14STf95S3cDstI5CiyvK1giLlbDw4ZThu
```

Install Dependencies:

```bash
pip install wandb lightning pytorch transformers
```

Then, you can start to train Siamese Network and Quadruplet Network，By following commands

```
python3 train.py --ood_non_vul_file=reveal_non_vul.jsonl \ 
	--ood_vul_file=reveal_vul.jsonl \
	--pretrained_model=model.bin \
	--train_file=adv_examples.jsonl \
	--batch_size=32 \
	--model_type=sia                        #quad for Quadruplet Network
```

Program will automatically divide `adv_examples.jsonl` into a training set and a validation set，And upload metrics and checkpoints to `wandb` Server, I strongly recommend you to use `wandb` to manage these data.

Once the training is done, you can use `InputReflector.ipynb` to inference, see the comments inside to know the details.



