import os
import sys
sys.path.append('..')
sys.path.append('../..')


from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch import loggers as pl_loggers

from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.model import HateDetection_LM_CNN
from src.preprocessing.constants import BASE_PATH
from src.preprocessing.utils import load_and_process, tokenize_and_prepare_dataset, get_args


def main(args):
    """
    Main function to run the Hate Speech Detection pipeline.

    This function performs the following steps:
    - Loads and processes tweets from a specified data path.
    - Splits the data into training, validation, and test sets.
    - Tokenizes the datasets and prepares them for training using a specified tokenizer.
    - Initializes a HateDetection_LM_CNN model with the specified parameters.
    - If training is enabled, trains the model with the training and validation data.
    - If testing is enabled, evaluates the model on the test data using a saved checkpoint.

    Parameters:
    args (argparse.Namespace): Command-line arguments containing the parameters for the script, which include
    the data path, model settings, training options, and other configuration settings.

    The script expects the following arguments:
    - data_path: Path to the CSV file containing the labeled data.
    - language_model_name_or_path: Pretrained language model to use.
    - model_checkpoint_path: Path to the checkpoint file for model evaluation.
    - do_train: Flag to indicate whether to train the model.
    - do_test: Flag to indicate whether to evaluate the model.
    - random_state: Seed for random number generation to ensure reproducibility.
    - max_epochs: The number of epochs to train the model.
    - train_batch_size: Batch size for the training data loader.
    - validation_batch_size: Batch size for the validation data loader.
    - max_length: Maximum sequence length for tokenization.
    - freeze_lm: Flag to indicate whether to freeze the language model during training.
    - number_of_classes: Number of classes for the classification task.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # specify which GPU(s) to be used

    tweets, lables = load_and_process(path=f"{BASE_PATH}/{args.data_path}")

    seed_everything(args.random_state)     # Set seed for reproducibility

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name_or_path)
    train_input, remaining_input, train_labels, remaining_labels = train_test_split(tweets, lables,
                                                                                     random_state=args.random_state, 
                                                                                     test_size=0.2,
                                                                                       stratify=lables)
    validation_input, test_input, validation_labels, test_labels = train_test_split(remaining_input,
                                                                                    remaining_labels,
                                                                                    random_state=args.random_state,
                                                                                    test_size=0.50,
                                                                                    stratify=remaining_labels)

    train_input_ids, train_attention_masks, train_labels = tokenize_and_prepare_dataset(sentences = train_input,
                                                                                         labels=train_labels,
                                                                                           tokenizer=tokenizer)
    train_data_loader = DataLoader(list(zip(train_input_ids, train_attention_masks, train_labels)),
                                    batch_size=args.train_batch_size,
                                    shuffle=True)

    validation_input_ids, validation_attention_masks, validation_labels = tokenize_and_prepare_dataset(
        sentences = validation_input,
        labels=validation_labels,
        tokenizer=tokenizer)
        
    validation_data_loader = DataLoader(list(zip(validation_input_ids, validation_attention_masks, validation_labels)),
                                        batch_size=args.validation_batch_size,
                                        shuffle=True)
    
    if args.class_weights:
      check_point_path = BASE_PATH / "checkpoints"/ f"{args.model_checkpoint_path}/{args.language_model_name_or_path}/class_weights/"
    else:
      check_point_path = BASE_PATH / "checkpoints" / f"{args.model_checkpoint_path}/{args.language_model_name_or_path}/no_class_weights/"
    model_checkpoint = ModelCheckpoint(
        dirpath= check_point_path,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    if args.class_weights:
       # Calculate the number of instances of each class
      class_counts = torch.bincount(torch.tensor(train_labels))

      # Calculate class weights inversely proportional to their frequency
      class_weights = 1. / class_counts.float()
    
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      class_weights.to(device)
    model = HateDetection_LM_CNN(language_model_name_or_path= args.language_model_name_or_path,
                                  max_length=args.max_length,
                                  freeze_lm=args.freeze_lm,
                                  number_of_classes=args.number_of_classes, 
                                  class_weights=class_weights if args.class_weights else None)
    if args.class_weights:
      logger_path = f"{BASE_PATH}/logs/{args.language_model_name_or_path}/class_weights/"
    else:
      logger_path = f"{BASE_PATH}/logs/{args.language_model_name_or_path}"
    tb_logger = pl_loggers.TensorBoardLogger(logger_path)
    
    if args.do_train:
      trainer = Trainer(max_epochs=args.max_epochs, callbacks=[model_checkpoint, EarlyStopping(monitor="val_loss", mode="min")], logger=tb_logger)
      trainer.fit(model, train_data_loader, validation_data_loader)
    
    if args.do_test:

      test_input_ids, test_attention_masks, test_labels = tokenize_and_prepare_dataset(
         sentences = test_input,
         labels=test_labels,
         tokenizer=tokenizer)
    
      test_data_loader = DataLoader(list(zip(test_input_ids, test_attention_masks, test_labels)),
                                        batch_size=args.validation_batch_size,
                                        shuffle=True)
      
      checkpoint_name = os.listdir(check_point_path)[0]
      checkpoint_path = os.path.join(check_point_path, checkpoint_name)
      model = HateDetection_LM_CNN.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                        language_model_name_or_path=args.language_model_name_or_path,
                                                        class_weights = class_weights if args.class_weights else None
                                                      )
      trainer = Trainer(logger=tb_logger)      
      trainer.test(model, test_data_loader)



if __name__ == "__main__":
  args = get_args()
  main(args)
    




