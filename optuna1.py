import Main

import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from optuna.integration.tensorboard import TensorBoardCallback


# tensorboard_callback = TensorBoardCallback("logs/optuna/", metric_name="accuracy")


# if os.path.exists(data['add_data']+'Optuna/') and os.path.isdir(data['add_data']+'Optuna/'):
#     shutil.rmtree(data['add_data']+'Optuna/')

# print(f"######################### TENSORBOARD #######################\ntensorboard --logdir={data['add_data'] + 'Optuna/'} --port 1374")

my_pruner = optuna.pruners.PercentilePruner(
    percentile=25, n_startup_trials=5, n_warmup_steps=5)
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=None)
study.optimize(Main.main, timeout=12*3600, n_trials=200)
