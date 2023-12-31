import random
import os
import cv2
import argparse
import os.path as osp
import numpy as np
import torch

from tqdm import tqdm

from tools import resize_bbox

from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

def parse_args():
    parser = argparse.ArgumentParser(
        description='Adversarial Attacked Image Generation pipeline with MMDetection')
    parser.add_argument(
        '--config',
        required=True, 
        help='model config file path')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='checkpoint file')
    parser.add_argument(
        '--img-dir',
        required=True,
        help='the directory to load the original images')
    parser.add_argument(
        '--save-dir',
        default=f'{os.getcwd()}/adv_images',
        help='the directory to save the generated images')
    parser.add_argument(
        '--method',
        default='DAG',
        help='the method to attack image')
    parser.add_argument(
        '--device',
        default='cuda')
    parser.add_argument(
        '--thr',
        default=0.5,
        help='threshold to control model results')
    parser.add_argument(
        '--is-save-detect',
        default=True,
        help='if it is true, it saves model inferece results')
    parser.add_argument(
        '--seed',
        default=322,
        help='random seed for adversarial labels')
    parser.add_argument(
        '--steps',
        default=200,
        help='steps for optimized method')
    parser.add_argument(
        '--lr',
        default=0.1,
        help='learning rate for optimized method')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # init variables
    CONFIG = args.config
    CHECKPOINT = args.checkpoint
    device = args.device
    img_path = args.img_dir
    output_path = args.save_dir
    if not os.path.exists(output_path): 
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path,'numpy'))
        os.makedirs(os.path.join(output_path,'detect'))

    extensions = ['.jpg', '.png', '.jpeg']

    model = init_detector(CONFIG,CHECKPOINT,device=device)

    for param in model.parameters():
        param.requires_grad = False
    
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    cfg = Config.fromfile(CONFIG)
    cfg.work_dir = osp.join('./work_dirs',osp.splitext(osp.basename(CONFIG))[0])

    files = [file for file in os.listdir(img_path) if any(file.endswith(ext) for ext in extensions)]

    img = cv2.imread(os.path.join(img_path,files[0]))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    pred = [inference_detector(model,img).detach().clone()] # 포맷 생성을 위한 tmp 생성
    random_labels = [i for i in range(len(model.dataset_meta['classes']))]
    random.seed(args.seed)
    random.shuffle(random_labels)

    epochs = int(args.steps)
    epsilon = 0.03

    if args.method == 'DAG':
        for file in files:
            model = model.eval()
            img = cv2.imread(os.path.join(img_path,file))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            pred = [inference_detector(model,img)]
            
            img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)
            img /= 255.0
            
            # 2. 해당 파일의 Pseudo GT를 모델 Prediction을 통해 만든다.
            model.forward(img,pred,mode='predict')
            pred[0] = pred[0].detach()

            # 3. 모델의 Pseudo GT의 label 값을 랜덤하게 바꿔 Fake GT를 만든다.

            ## score가 0.7 이상인 예측값들로 Pseudo GT를 생성
            filtered_labels = pred[0].pred_instances['labels'][pred[0].pred_instances['scores'] > 0.7]
            filtered_labels = filtered_labels.cpu().apply_(lambda x: random_labels[x]).cuda() # 랜덤하게 label을 변경
            filtered_bboxes = pred[0].pred_instances['bboxes'][pred[0].pred_instances['scores'] > 0.7]

            tmp_data = InstanceData(metainfo=pred[0].metainfo)
            tmp_data.labels = filtered_labels
            tmp_data.bboxes = filtered_bboxes
            pred[0].gt_instances = tmp_data.clone() # Fake GT를 pred에 적용
            del(pred[0]['pred_instances']) # pred_instances 제거

            # 4. Fake GT를 통해 Adversarial Attack 이미지 생성
            img.requires_grad = True
            origin_img = img.detach().clone()
            optimizer = torch.optim.SGD([img],lr=args.lr)

            with tqdm(total=epochs) as pbar:
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    normal_losses = model.forward(img,pred,mode='loss')
                    loss = normal_losses['loss_cls']
                    pbar.set_postfix(loss=f"{loss:.4f}")
                    pbar.update(1)
                    loss.backward()
                    optimizer.step()
                    
                    with torch.no_grad():
                        pertubation = torch.clamp(img[0].detach()-origin_img[0],min=-epsilon,max=epsilon)
                        img.data = origin_img.detach() + pertubation.detach()
                        img.clamp_(0,1)
                    
                    
            del(optimizer)
            del(loss)
            del(normal_losses)

            img = img.detach()
            np.save(os.path.join(output_path,f"{file.split('.')[0]}.npy"),img.to('cpu').numpy())

            if args.is_save_detect == True:
                # 5. Adversarial Attack 된 이미지의 inference 결과를 저장
                model.forward(img,pred,mode='predict')

                ## 모델의 feed forward 결과가 config 파일의 resize 크기에 맞춰져 나오기 때문에 이를 다시 이미지 사이즈로 resize 함
                for idx, bbox in enumerate(pred[0].pred_instances['bboxes']):
                    new_bbox = resize_bbox(bbox,pred[0].ori_shape,pred[0].img_shape)
                    pred[0].pred_instances['bboxes'][idx] = new_bbox
                
                visualizer.add_datasample(
                    'result',
                    (img[0].cpu().permute(1,2,0).numpy()*255).astype('uint8'),
                    data_sample=pred[0],
                    draw_gt=False,
                    pred_score_thr = float(args.thr),
                    wait_time=0,
                    out_file = os.path.join(output_path,'detect',file)
                )

if __name__ == '__main__':
    main()
