def log_result(epoch, avg_train_loss, avg_eval_loss, train_metrics, eval_metrics):
    print(f"Epoch {epoch}:")
    print(f"Train Loss: {avg_train_loss}")
    for key in train_metrics:
        for metric in train_metrics[key]:
            print(f"train-{key}-{metric}: {train_metrics[key][metric]}")

    print()
    print(f"Eval Loss: {avg_eval_loss}")
    for key in eval_metrics:
        for metric in eval_metrics[key]:
            print(f"eval-{key}-{metric}: {eval_metrics[key][metric]}")
        
    print()
