# import pandas as pd
# import torch.nn.functional as Ftorch
# from torch.utils.data import DataLoader
# import os
# import cv2
# from tqdm import *
#
# from models import *
# from augmentation import *
# from dataset import FrameDataset, RAFDataset
#
#
# device = torch.device('cuda')
#
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
#
#
# def denorm(tensor):
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)
#     return tensor
#
#
# def combine_heatmap_with_image(image, heatmap):
#     heatmap = heatmap - np.min(heatmap)
#     if np.max(heatmap) != 0:
#         heatmap = heatmap / np.max(heatmap)
#     heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))
#
#     scaled_image = denorm(image) * 255
#     scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))
#
#     cam = heatmap + np.float32(scaled_image)
#     cam = cam - np.min(cam)
#     if np.max(cam) != 0:
#         cam = cam / np.max(cam)
#
#     heat_map = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)
#     return heat_map
#
#
# def predict(model, loader, outheatmap):
#     model.eval()
#     logits = []
#     with torch.no_grad():
#         for dct in tqdm(loader, total=len(loader)):
#             images = dct['images'].to(device)
#             targets = dct['targets'].to(device)
#             image_names = dct['image_names']
#             pred = model(images, targets)
#             if isinstance(model, GAIN):
#                 logit, output_am, heatmap = pred
#             elif isinstance(model, GCAM):
#                 logit, heatmap = pred
#             elif isinstance(model, GAINMask):
#                 logit, output_am, heatmap, mask = pred
#             else:
#                 logit = pred
#             logit = Ftorch.softmax(logit)
#             logit = logit.detach().cpu().numpy()
#             logits.append(logit)
#
#             for image, ac, image_name in zip(images, heatmap, image_names):
#                 ac = ac.data.cpu().numpy()[0]
#                 heat_map = combine_heatmap_with_image(
#                     image=image,
#                     heatmap=ac
#                 )
#                 cv2.imwrite(f"{outheatmap}/{image_name}", heat_map)
#
#     preds = np.concatenate(logits, axis=0)
#     return preds
#
#
# def predict_all():
#     test_csv = "/media/ngxbac/Bac/competition/emotiw/notebook/RAF/csv/test.csv"
#     log_dir = f"/media/ngxbac/DATA/logs_omg/rafdb/gain_mask/mask_select_mse/"
#     outheatmap = f"{log_dir}/best_heatmaps/"
#     os.makedirs(outheatmap, exist_ok=True)
#
#     model = GAINMask(grad_layer='layer4', num_classes=7)
#
#     checkpoint = f"{log_dir}/checkpoints/best.pth"
#     checkpoint = torch.load(checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()
#
#     # Dataset
#     dataset = RAFDataset(
#         df_path=test_csv,
#         transform=valid_aug(224),
#         mode="train"
#     )
#
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=32,
#         shuffle=False,
#         num_workers=4,
#     )
#
#     pred = predict(model, loader, outheatmap)
#
#     # pred = np.asarray(pred).mean(axis=0)
#     all_preds = np.argmax(pred, axis=1)
#     df = pd.read_csv(test_csv)
#     gt = df.label.values - 1
#     from sklearn.metrics import accuracy_score
#     acc = accuracy_score(gt, all_preds)
#     print(acc)
#     submission = df.copy()
#     submission['label'] = all_preds.astype(int)
#     submission.to_csv(f'{log_dir}/prediction.csv', index=False)
#     np.save(f"{log_dir}/prediction.npy", pred)
#
#
# if __name__ == '__main__':
#     predict_all()
