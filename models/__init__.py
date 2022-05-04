from .ViT import *
# import importlib
# import os

# # Automately import the module defined in this folder
# models_dir = os.path.dirname(__file__)
# for file in os.listdir(models_dir):
#     path = os.path.join(models_dir, file)
#     if (
#         not file.startswith('_')
#         and not file.startswith('.')
#         and (file.endswith('.py') or os.path.isdir(path))
#     ):
#         model_name = file[: file.find('.py')] if file.endswith('.py') else file
#         module = importlib.import_module(f'models.{model_name}')
