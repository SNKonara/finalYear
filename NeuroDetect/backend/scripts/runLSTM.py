import argparse
import pandas as pd
from LSTModel.preprocessing import Preprocessor
from LSTModel.trainer import LSTMTrainer
from LSTModel.evaluator import ModelEvaluator
from LSTModel.utils import save_model, set_random_seeds

def main(args):
    set_random_seeds(42)

    # Load data (CSV must contain 'is_fraud' column)
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} rows from {args.data_path}")

    # Prepare sequences and labels
    pre = Preprocessor(sequence_length=args.sequence_length, fraud_boost_factor=args.fraud_boost_factor)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = pre.prepare_improved_lstm_data(df)

    # Scale
    X_train_scaled, X_val_scaled, X_test_scaled = pre.scale_data(X_train, X_val, X_test)

    # Model & training params
    model_params = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "use_attention": args.use_attention
    }
    training_params = {
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay
    }

    # Train
    trainer = LSTMTrainer(device=None)
    metrics = trainer.train(X_train_scaled, y_train, X_val_scaled, y_val, feature_names,
                            model_params=model_params, training_params=training_params)

    # Evaluate on test set
    evaluator = ModelEvaluator(trainer.model, device=None)
    results = evaluator.evaluate(X_test_scaled, y_test, pre.scaler, threshold_tuning=True)

    # Save model + metadata
    save_model(trainer.model, pre.scaler, feature_names, results, 
               {**model_params, **training_params}, filename=args.output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate FraudLSTM")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV with 'is_fraud' column")
    parser.add_argument("--output-model", type=str, default="lstm_model.pth", help="Model file to save")
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--fraud-boost-factor", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--use-attention", type=bool, default=True)

    args = parser.parse_args()
    main(args)