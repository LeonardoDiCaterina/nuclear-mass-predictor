import functools

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_summary_metrics,
    evaluate_all_models,
    scale_features,
    split_historical_data,
    train_jax_model,
    train_pytorch_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        node(
            func=split_historical_data,
            inputs=["primary_features", "params:zeng_2022_training"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_historical_data_node",
        ),
        node(
            func=scale_features,
            inputs=["X_train", "X_test", "y_train", "y_test", "params:zeng_2022_training"],
            outputs=["X_train_scaled", "X_test_scaled", "y_train_scaled", "y_test_scaled", "scaler_params"],
            name="scale_features_node",
        ),
    ]

    model_names = ["ANN7"]
    eval_inputs = {
        "X_test_scaled": "X_test_scaled",
        "y_test": "y_test",
        "X_test_raw": "X_test",
        "scaler_params": "scaler_params",
        "params": "params:zeng_2022_training"
    }

    for mn in model_names:
        pt_func = functools.partial(train_pytorch_model, model_name=mn)
        functools.update_wrapper(pt_func, train_pytorch_model)
        
        jax_func = functools.partial(train_jax_model, model_name=mn)
        functools.update_wrapper(jax_func, train_jax_model)
        
        # PyTorch Node
        nodes.append(
            node(
                func=pt_func,
                inputs={
                    "X_train_scaled": "X_train_scaled", 
                    "y_train_scaled": "y_train_scaled", 
                    "scaler_params": "scaler_params",
                    "params": "params:zeng_2022_training"
                },
                outputs=[f"{mn}_pytorch_model", f"{mn}_pytorch_loss_history"],
                name=f"train_{mn}_pytorch_model_node"
            )
        )
        # JAX Node
        nodes.append(
            node(
                func=jax_func,
                inputs={
                    "X_train_scaled": "X_train_scaled", 
                    "y_train_scaled": "y_train_scaled", 
                    "scaler_params": "scaler_params",
                    "params": "params:zeng_2022_training"
                },
                outputs=[f"{mn}_jax_model", f"{mn}_jax_loss_history"],
                name=f"train_{mn}_jax_model_node"
            )
        )
        
        # Add to evaluation inputs
        eval_inputs[f"{mn}_pytorch"] = f"{mn}_pytorch_model"
        eval_inputs[f"{mn}_jax"] = f"{mn}_jax_model"

    nodes.append(
        node(
            func=evaluate_all_models,
            inputs=eval_inputs,
            outputs="zeng_2022.unified_test_predictions",
            name="evaluate_all_models_node",
        )
    )

    nodes.append(
        node(
            func=compute_summary_metrics,
            inputs="zeng_2022.unified_test_predictions",
            outputs="zeng_2022.evaluation_metrics",
            name="compute_summary_metrics_node",
        )
    )

    return pipeline(nodes)
