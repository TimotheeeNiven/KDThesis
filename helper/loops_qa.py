import sys
import time
import torch
import torch.nn as nn
import evaluate
from transformers import AutoTokenizer

from .util import AverageMeter

def train_squad(epoch, train_loader, model, loss_fn, optimizer, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_tracker = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_ids = batch['input_ids'].long().cuda()
        attention_mask = batch['attention_mask'].cuda()
        start_positions = batch['start_positions'].cuda()
        end_positions = batch['end_positions'].cuda()

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if hasattr(model, "supports_token_type") and model.supports_token_type:
            kwargs["token_type_ids"] = batch["token_type_ids"].long().cuda()

        optimizer.zero_grad()
        output = model(**kwargs)
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            start_logits, end_logits = output
        else:
            _, (start_logits, end_logits) = output

        loss = nn.CrossEntropyLoss()(start_logits, start_positions) + nn.CrossEntropyLoss()(end_logits, end_positions)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input_ids.size(0))

        pred_start = torch.argmax(start_logits, dim=1)
        pred_end = torch.argmax(end_logits, dim=1)
        correct = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
        acc_tracker.update(correct / input_ids.size(0), input_ids.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc_tracker))
            sys.stdout.flush()

    return acc_tracker.avg, losses.avg

def train_squad_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    model_s = module_list[0]
    model_t = module_list[-1]
    model_s.train()
    model_t.eval()

    criterion_ce = criterion_list[0]
    criterion_kd = criterion_list[1]
    criterion_div = criterion_list[2] if len(criterion_list) > 2 else None

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_tracker = AverageMeter()

    device = next(model_s.parameters()).device
    end = time.time()

    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_ids = batch['input_ids'].long().to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if hasattr(model_s, "supports_token_type") and model_s.supports_token_type:
            kwargs["token_type_ids"] = batch["token_type_ids"].long().to(device)

        feat_s, (start_logits_s, end_logits_s) = model_s(**kwargs)

        if hasattr(model_t, "supports_token_type") and model_t.supports_token_type:
            kwargs["token_type_ids"] = batch["token_type_ids"].long().to(device)
        else:
            kwargs.pop("token_type_ids", None)

        with torch.no_grad():
            feat_t, (start_logits_t, end_logits_t) = model_t(**kwargs)
            feat_t = [f.detach() for f in feat_t]

        loss_ce = criterion_ce(start_logits_s, start_positions) + criterion_ce(end_logits_s, end_positions)

        loss_kd = 0
        if opt.distill == 'kd':
            T = opt.kd_T
            kd_loss_start = criterion_kd(
                nn.functional.log_softmax(start_logits_s / T, dim=1),
                nn.functional.softmax(start_logits_t / T, dim=1)
            ) * (T * T)
            kd_loss_end = criterion_kd(
                nn.functional.log_softmax(end_logits_s / T, dim=1),
                nn.functional.softmax(end_logits_t / T, dim=1)
            ) * (T * T)
            loss_kd = (kd_loss_start + kd_loss_end) / 2

        elif opt.distill == 'attention':
            # Get intermediate features excluding embeddings and final
            g_s_raw = feat_s[1:-1]
            g_t_raw = feat_t[1:-1]

            # Align last k layers (default 4, adjust if needed)
            k = min(4, len(g_s_raw), len(g_t_raw))
            g_s_selected = g_s_raw[-k:]
            g_t_selected = g_t_raw[-k:]

            # Build regressors on-the-fly
            regressors = [
                nn.Linear(s.shape[-1], t.shape[-1]).to(device) if s.shape[-1] != t.shape[-1] else nn.Identity().to(device)
                for s, t in zip(g_s_selected, g_t_selected)
            ]

            # Apply regression and compute loss
            g_s = [reg(s) for reg, s in zip(regressors, g_s_selected)]
            g_s = [nn.functional.normalize(f, dim=-1) for f in g_s]
            g_t = [nn.functional.normalize(f, dim=-1) for f in g_t_selected]

            loss_kd = sum([criterion_kd(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

        elif opt.distill == 'rkd':
            # Use intermediate hidden states (excluding embeddings and final)
            g_s_raw = feat_s[1:-1]
            g_t_raw = feat_t[1:-1]

            # Align last k layers (e.g., 4)
            k = min(4, len(g_s_raw), len(g_t_raw))
            g_s_selected = g_s_raw[-k:]
            g_t_selected = g_t_raw[-k:]

            # Build projection layers (lazy init, similar to attention)
            if not hasattr(opt, 'rkd_proj_list'):
                opt.rkd_proj_list = nn.ModuleList([
                    nn.Linear(s.shape[-1], t.shape[-1]).to(s.device) if s.shape[-1] != t.shape[-1] else nn.Identity().to(s.device)
                    for s, t in zip(g_s_selected, g_t_selected)
                ])
                # Add parameters to optimizer
                optimizer.add_param_group({'params': opt.rkd_proj_list.parameters()})

            # Apply projection + normalize (for distance-based loss)
            g_s = [nn.functional.normalize(reg(s[:, 0, :]), dim=-1) for reg, s in zip(opt.rkd_proj_list, g_s_selected)]
            g_t = [nn.functional.normalize(t[:, 0, :], dim=-1) for t in g_t_selected]

            # RKD loss summed over selected layers
            loss_kd = sum([criterion_kd(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])

        else:
            raise NotImplementedError(f"Distillation method '{opt.distill}' is not supported.")

        loss = opt.alpha * loss_ce + opt.beta * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input_ids.size(0))

        pred_start = torch.argmax(start_logits_s, dim=1)
        pred_end = torch.argmax(end_logits_s, dim=1)
        correct = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
        acc_tracker.update(correct / input_ids.size(0), input_ids.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc_tracker))
            sys.stdout.flush()

    return acc_tracker.avg, losses.avg

def validate_squad(val_loader, model, loss_fn, opt):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_tracker = AverageMeter()

    squad_metric = evaluate.load("squad")
    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer)
    all_predictions = []
    all_references = []

    device = next(model.parameters()).device

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].long().to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if hasattr(model, "supports_token_type") and model.supports_token_type:
                kwargs["token_type_ids"] = batch["token_type_ids"].long().to(device)

            _, (start_logits, end_logits) = model(**kwargs)

            loss = nn.CrossEntropyLoss()(start_logits, start_positions) + nn.CrossEntropyLoss()(end_logits, end_positions)
            losses.update(loss.item(), input_ids.size(0))

            pred_start = torch.argmax(start_logits, dim=1)
            pred_end = torch.argmax(end_logits, dim=1)
            correct = ((pred_start == start_positions) & (pred_end == end_positions)).sum().item()
            acc_tracker.update(correct / input_ids.size(0), input_ids.size(0))

            for i in range(input_ids.size(0)):
                start_idx = pred_start[i].item()
                end_idx = pred_end[i].item()

                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx

                offset_mapping = batch['offset_mapping'][i]
                context = batch['context'][i]

                if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
                    prediction_text = ""
                else:
                    start_char = offset_mapping[start_idx][0]
                    end_char = offset_mapping[end_idx][1]
                    prediction_text = context[start_char:end_char]

                gold_answer = batch['answers'][i]
                if "text" not in gold_answer or not gold_answer["text"]:
                    gold_answer = {"text": [""], "answer_start": [0]}

                all_predictions.append({
                    "id": batch['id'][i],
                    "prediction_text": prediction_text
                })
                all_references.append({
                    "id": batch['id'][i],
                    "answers": gold_answer
                })

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses))

    result = squad_metric.compute(predictions=all_predictions, references=all_references)
    print(f' * EM {result["exact_match"]:.2f} F1 {result["f1"]:.2f} Loss {losses.avg:.4f}')

    return result["exact_match"], result["f1"], losses.avg
