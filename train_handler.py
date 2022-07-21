import torch
import os
import numpy as np
import time

from model.utils import DiceLoss, Logger, AverageMeter
from config import *
# from utils.data_handler import make_loader


class TrainHandler:

    def __init__(self, model, train_loader, valid_loader, optimizer, scheduler, config):
        self.model=model
        self.optimizer = optimizer
        self.scheduler=scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.folder_name = MODEL_OUTPUT_DIR

        self.logger = Logger(os.path.join(MODEL_OUTPUT_DIR, 'log.txt'))

    def run(self):

        result_list = []
        self.logger.log('Training model')
        for fold in range(self.config['num_folds']):
            self.logger.log('----')
            self.logger.log(f'FOLD: {fold}')

            result_dict = self.run_fold(model = self.model,
                                        train_loader = self.train_loader,
                                        valid_loader = self.valid_loader,
                                        optimizer = self.optimizer,
                                        scheduler = self.scheduler,
                                        batch_size=self.config['batch_size'],
                                        fold=fold,
                                        model_ouput_location=self.folder_name,
                                        epochs=self.config['epochs'],
                                        evaluate_interval_fraction=self.config['evaluate_interval'],
                                        strict=True)
            result_list.append(result_dict)
            self.logger.log('----')

        [self.logger.log("FOLD::" + str(i) + "Loss:: " + str(fold['best_val_loss'])) for i, fold in
             enumerate(result_list)]

        self.logger.save_log()


    def run_fold(self, model, train_loader, valid_loader, optimizer, scheduler,batch_size, fold=0,
                 model_ouput_location="", epochs=1, evaluate_interval_fraction=1, strict=True):



        result_dict = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': np.inf
        }


        trainer = Trainer(model, optimizer, scheduler, model_output_location=model_ouput_location, logger=self.logger,
                          evaluate_interval_fraction=evaluate_interval_fraction,
                          config=self.config)

        
        train_time_list = []

        for epoch in range(epochs):
            # adjust_learning_rate(optimizer, epoch, 0.1)

            result_dict['epoch'] = epoch

            torch.cuda.synchronize()
            tic1 = time.time()

            result_dict = trainer.train(train_loader, valid_loader, epoch, result_dict, fold)

            torch.cuda.synchronize()
            tic2 = time.time()
            train_time_list.append(tic2 - tic1)

        torch.cuda.empty_cache()
        del model, optimizer, train_loader, valid_loader

        return result_dict


class Trainer:
    def __init__(self, model, optimizer, scheduler,
                 model_output_location, logger, config, log_interval=1,
                 evaluate_interval_fraction=1):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.evaluate_interval_fraction = evaluate_interval_fraction
        self.evaluator = Evaluator(self.model)
        self.model_output_location = model_output_location
        self.logger = logger

    def train(self, train_loader, valid_loader, epoch, result_dict, fold):
        count = 0
        losses = AverageMeter()

        self.model.train()

        evaluate_interval = int((len(train_loader) - 1) * self.evaluate_interval_fraction)
        for batch_idx, batch_data in enumerate(train_loader):
            image, mask, target_ind = batch_data['image'], batch_data['mask'], batch_data['target_ind']
            image, mask, target_ind = image.cuda(), mask.cuda(), target_ind.cuda()

            mask_pred = self.model.forward(image)
            mask_pred = torch.sigmoid(mask_pred)
            loss = DiceLoss(mask, mask_pred, target_ind)
            count += target_ind.size(0)
            losses.update(loss.item(), target_ind.size(0))  # ------ may need to change this ?

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()

            if batch_idx % self.log_interval == 0:
                _s = str(len(str(len(train_loader.sampler))))

                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count,
                                                                               len(train_loader.sampler),
                                                                               100 * count / len(
                                                                                   train_loader.sampler)),
                    'train_loss: {: >4.5f}'.format(losses.avg),
                ]

                self.logger.log(', '.join(ret))

            if batch_idx != 0 and batch_idx % evaluate_interval == 0:
                result_dict = self.evaluator.evaluate(
                    valid_loader,
                    epoch,
                    result_dict
                )
                if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
                    self.logger.log("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch,
                                                                                                     result_dict[
                                                                                                         'val_loss'][
                                                                                                         -1]))
                    result_dict["best_val_loss"] = result_dict['val_loss'][-1]
                    torch.save(self.model.state_dict(), os.path.join(self.model_output_location, f"model{fold}.bin"))
                self.model.train()
        result_dict['train_loss'].append(losses.avg)
        self.scheduler.step()
        return result_dict


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader, epoch, result_dict):
        losses = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                image, mask, target_ind = batch_data['image'], batch_data['mask'], batch_data['target_ind']
                image, mask, target_ind = image.cuda(), mask.cuda(), target_ind.cuda()

                mask_pred = self.model.forward(image)
                mask_pred = torch.sigmoid(mask_pred)
                loss =  DiceLoss(mask, mask_pred,target_ind)
                # print(loss)
                losses.update(loss.item(), target_ind.size(0))

        print('----Validation Results Summary----')
        print('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, losses.avg))

        result_dict['val_loss'].append(losses.avg)
        return result_dict
