import torch
import time
import numpy as np
from src.pytorchtools import EarlyStopping
import gc
from src.helpers import format_time
import torch.nn.functional as F
from transformers import BertTokenizer
from src.features.preprocess_feature_creation import create_dataloaders_BERT, nrc_feats, vad_feats
from src.evaluate import monitor_metrics


def validation(model, val_dataloader, device, loss_fn, vad_nrc=False, tokenizer=None, MAX_LEN=None,
               project_root_path=None):
    """
    After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_loss = []
    logits_all = []
    b_labels_all = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        if vad_nrc:
            lex_feats = nrc_feats(b_input_ids, tokenizer).to(device)
            vad = vad_feats(b_input_ids, tokenizer, MAX_LEN, project_root_path).to(device)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask, lex_feats, vad)
        else:
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

        # Compute loss

        b_labels = b_labels.type(torch.LongTensor).to(device)
        # b_labels = b_labels.float().to(self.device)
        loss = loss_fn(logits, b_labels)

        val_loss.append(loss.item())

        # Get the predictions
        logits_all.append(logits)
        b_labels_all.append(b_labels)

    logits_all = torch.cat(logits_all, dim=0)
    b_labels_all = torch.cat(b_labels_all, dim=0)
    val_accuracy, val_f1 = monitor_metrics(logits_all, b_labels_all)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)

    return val_loss, val_accuracy, val_f1


def train(model, train_dataloader, EPOCHS, es, patience, project_root_path, device, loss_fn,
          val_dataloader, optimizer, scheduler, evaluation=False, vad_nrc=False, tokenizer=None, MAX_LEN=None):
    """
    Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n", flush=True)
    early_stopping = EarlyStopping(metric=es, patience=patience, verbose=True,
                                   path=project_root_path + '/models/checkpoint.pt')
    t0 = time.time()
    for epoch_i in range(EPOCHS):
        gc.collect()
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9}"
            f"| {'Elapsed':^12} | {'Elapsed Total':^12}",
            flush=True)
        print("-" * 100, flush=True)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            if vad_nrc:
                lex_feats = nrc_feats(b_input_ids, tokenizer).to(device)
                vad = vad_feats(b_input_ids, tokenizer, MAX_LEN, project_root_path).to(device)

                # Zero out any previously calculated gradients
                model.zero_grad()
                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask, lex_feats, vad)
            else:
                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            b_labels = b_labels.type(torch.LongTensor).to(device)
            # b_labels = b_labels.float().to(self.device)
            loss = loss_fn(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed_batch = format_time(time.time() - t0_batch)
                time_elapsed_total = format_time(time.time() - t0_epoch)
                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10}"
                    f"| {'-':^9}|{time_elapsed_batch:^12} | {time_elapsed_total:^12}",
                    flush=True)

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 100)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy, val_f1 = validation(model, val_dataloader, device, loss_fn, vad_nrc,
                                                        tokenizer, MAX_LEN, project_root_path)

            # Print performance over the entire training data
            time_elapsed = format_time(time.time() - t0_epoch)
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} "
                f"|  {'Elapsed':^12} | {'Elapsed Total':^12}",
                flush=True)
            print("-" * 100, flush=True)
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} "
                f"| {'-':^12}| {time_elapsed:^12}",
                flush=True)
            print("-" * 100)

            if es == 'f1':
                early_stopping(val_f1, model)
            elif es == 'loss':
                early_stopping(val_loss, model)
            else:
                torch.save(model.state_dict(), project_root_path + '/models/checkpoint.pt')

            if early_stopping.early_stop:
                print("Early stopping")
                break
        print("\n")
        model.load_state_dict(torch.load(project_root_path + '/models/checkpoint.pt'))

    print(f"Total training time: {format_time(time.time() - t0)}", flush=True)

    print("Training complete!", flush=True)


def predict(X_test, MAX_LEN, BATCH_SIZE, device, vad_nrc=False, model_path=None, model_name=None, project_root_path=None):
    """
    Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    t0 = time.time()
    model = torch.jit.load(model_path+model_name + '.pt', map_location=torch.device(device))
    # model = torch.jit.load(project_root_path + '/models/model_scripted_BERT_simple.pt')

    model.eval()

    y_test = None

    tokenizer = BertTokenizer.from_pretrained(model_path + "/tokenizer/")
    test_dataloader = create_dataloaders_BERT(X_test, y_test, tokenizer, MAX_LEN, BATCH_SIZE,
                                              sampler='sequential', token_type=False)
    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        if vad_nrc:
            lex_feats = nrc_feats(b_input_ids, tokenizer).to(device)
            vad = vad_feats(b_input_ids, tokenizer, MAX_LEN, project_root_path).to(device)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask, lex_feats, vad)
        else:
            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)

    return probs
