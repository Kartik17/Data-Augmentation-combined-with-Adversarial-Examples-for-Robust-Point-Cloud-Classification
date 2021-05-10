import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#from pointnet2_cls_msg import PointNet2
#from point_augment import Augmentor
import utils.train_utils
from models.model import ClassificationPointNet
from models.dgcnn import DGCNN_cls
from models.fgsm import fgsm_attack
import os
import tqdm
import logging
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PointAugment.Common import data_utils as d_utils
import random
from PointAugment.Common import loss_utils
from PointAugment.Common.ModelNetDataLoader import ModelNetDataLoader
from PointAugment.Augment.pointnet import PointNetCls
from PointAugment.Augment.augmentor import Augmentor
import sklearn.metrics as metrics
debug = True

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
            
class PointNetTrain():
    def __init__(self, model_config, device):
                
        self._device          = device
        self._categories     = model_config['NUM_CLASSES']
        self._model           = DGCNN_cls(num_classes = self._categories).to(self._device) #PointNetCls(self._categories).to(self._device) #PointNetClassifier(num_classes = model_config['TRAINING']['NUM_CLASSES']).to(self._device)
        self._epochs          = model_config['TRAINING']['EPOCHS']
        self._criterion_name  = model_config['TRAINING']['CRITERION']
        self._optimizer_name  = model_config['TRAINING']['OPTIMIZER']
        self._lr              = model_config['TRAINING']['LR']
        self._augmentor_name  = model_config['AUGMENTOR']['NAME']
        self._alpha = model_config['AUGMENTOR']['ALPHA']
        self._augmentor = None
        self._path = model_config['TRAINING']['PATH']

        #self.weight = weights.to(self._device).float()

        if(debug == True):
            print("Device: {}".format(self._device))
            print("Epochs: {}".format(self._epochs))
            print("Criterion: {}".format(self._criterion_name))
            print("Optimizer: {}".format(self._optimizer_name))
            print("Learning Rate: {}".format(self._lr))

        if(self._augmentor_name == 'FGSM'):
            
            self._eps = model_config['AUGMENTOR']['EPS']
            self._augmentor = fgsm_attack
        elif(self._augmentor_name == 'PA'):
            pass

        try:
            if(self._criterion_name == "CrossEntropyLoss"):
                self._criterion = nn.CrossEntropyLoss()
            else:
                self._criterion = nn.CrossEntropyLoss()
        except:
            print("Criterion invalid .... Proceedding with default CrossEntropyLoss")
            self._criterion = nn.CrossEntropyLoss()
        else:
            print("Criterion is {}".format(self._criterion_name))


        try:
            if(self._optimizer_name == "Adam"):
                self._optimizer = optim.Adam(self._model.parameters(), lr = self._lr)
            else:
                self._optimizer = optim.Adam(self._model.parameters(), lr = self._lr)
        except:
            print("Optimizer invalid .... Proceedding with default Adam")
            self._optimizer = optim.Adam(self._model.parameters(), lr = self._lr)
        else:
            print("Optimizer is {}".format(self._optimizer_name))
    

        self._lr_scheduler_classifier = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size = model_config['TRAINING']['LR_SCHEDULER']['STEP_SIZE'], 
                                        gamma = model_config['TRAINING']['LR_SCHEDULER']['GAMMA'])

    def train(self, trainloader, validloader, adv = False, save_flag = True):
        if(adv):
            self.adv_train_model(trainloader, validloader)
        else:
            self.train_model(trainloader, validloader)
        

    def test(self, validloader, eps_list = [0.3,0.6,0.9,1.2,1.5], adv = True):
        try:
            self._model.load_state_dict(torch.load(self._path))
        except:
            print("Cannot Load the saved Model")

        #self._model.eval()
        if adv == True:
            for eps in eps_list:
                self._eps = eps
                self.adv_valid_model(validloader)
        else:
            self.valid_model(validloader)

    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, epochs):
        if epochs > 0:
            self._epochs = epochs
        else:
            print("Enter valid value of greater than zero. ")

    def save_model(self):
        torch.save(self._model.state_dict(), self._path)


    def train_model(self, trainloader, validloader, path = "./saved_model.pth", print_epoch = 5, save_flag = True):
        
        for epoch in range(self._epochs):
            
            loss_tot = 0.0
            total_examples, accurate_examples = 0, 0

            for batch_idx, batch_data in enumerate(trainloader):
                self._model.train()
                self._optimizer.zero_grad()
                data   = batch_data[0].type('torch.FloatTensor').to(self._device).transpose(2,1)
                labels = batch_data[1].type('torch.FloatTensor').to(self._device).view(-1)
                
                outputs, feature_transform = self._model(data)
                loss = self._criterion(outputs, labels.long())
                loss_tot += loss.item()

                

                pred_label = torch.argmax(outputs, dim=1)
                accurate_examples += np.sum(pred_label.cpu().numpy() == labels.cpu().numpy())
                total_examples += len(pred_label.cpu().numpy())

                loss.backward()
                self._optimizer.step()
                from fgsm import visualize_batch
                
            
            print("Loss per Epoch - {}: {}, Train Accuracy : {}".format(epoch, loss_tot, (accurate_examples/ total_examples)))
            self._lr_scheduler_classifier.step()
            if((epoch+1) % print_epoch == 0):
                self.valid_model(validloader, epoch)

    def adv_train_model(self, trainloader, validloader, path = "./saved_model.pth", print_epoch = 5, save_flag = True):
        current_best_accuracy = 0.0
        for epoch in range(self._epochs):
            self._model.train()
            mean_correct, test_pred, test_true = [], [], []
            for batch_idx, batch_data in enumerate(trainloader):
                
                self._optimizer.zero_grad() 
                
                points = batch_data[0].permute(0,2,1).to(self._device)
                labels = batch_data[1].to(self._device)
                  
                pred_pc, feature_transform, _ = self._model(points)
                loss = self._criterion(pred_pc, labels.long())
                
                loss.backward()
                self._optimizer.step()

                self._optimizer.zero_grad() 
                if(self._augmentor_name == 'FGSM'):
                    aug_data = self._augmentor(self._model, self._criterion, points, labels, self._eps)
                
                outputs, feature_transform = self._model(aug_data)
                loss = self._criterion(outputs, labels.long())
                
                loss.backward()
                self._optimizer.step()                    
                
                test_true.append(labels.cpu().numpy())
                test_pred.append(torch.argmax(pred_pc, dim = 1).detach().cpu().numpy())
                pred_pc = torch.argmax(pred_pc, dim = 1)
                correct = pred_pc.eq(labels.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)

            print("Accuracy: {}".format(test_acc))
            self._lr_scheduler_classifier.step()
            if((epoch+1) % print_epoch == 0):
                val_accuracy = self.valid_model(validloader, epoch)
                if(val_accuracy > current_best_accuracy):
                    current_best_accuracy = val_accuracy
                    self.save_model()
                print("Valid Accuracy {}".format(val_accuracy))
                
        return

    def valid_model(self, validloader, epoch = 0):
        #self._model.eval()
        mean_correct, test_pred, test_true = [], [], []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(validloader):
                points   = batch_data[0].permute(0,2,1).to(self._device)
                labels   = batch_data[1].to(self._device)
                pred_pc, _  = self._model(points)
                
                test_true.append(labels.cpu().numpy())
                test_pred.append(torch.argmax(pred_pc, dim = 1).detach().cpu().numpy())
                pred_pc = torch.argmax(pred_pc, dim = 1)
                correct = pred_pc.eq(labels.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            valid_acc = metrics.accuracy_score(test_true, test_pred)
        
        print("Valid Accuracy {}".format(valid_acc))
        return valid_acc


    def adv_valid_model(self, validloader, epoch = 0):        
        total_examples, accurate_examples, accurate_aug_examples = 0, 0, 0
        #self._model.eval()
        for batch_idx, batch_data in enumerate(validloader):
            
            with torch.no_grad():
                labels = batch_data[1].to(self._device)
                data = batch_data[0].permute(0,2,1).to(self._device)
                outputs, feature_transform  = self._model(data)
                pred_label = torch.argmax(outputs, dim=1)

            if(self._augmentor_name == 'FGSM'):
                aug_data = self._augmentor(self._model, self._criterion, data, labels, self._eps)
            
            outputs, feature_transform = self._model(aug_data)
            pred_label_aug = torch.argmax(outputs, dim=1)                

            accurate_aug_examples += np.sum(pred_label_aug.cpu().numpy() == labels.cpu().numpy())
            accurate_examples += np.sum(pred_label.cpu().numpy() == labels.cpu().numpy())
            total_examples += len(pred_label.cpu().numpy())

            #visualize_batch(data, pred_label, labels, categories)
        
        try:
            accuracy_clean  = accurate_examples/total_examples
            accuracy_adv = accurate_aug_examples/total_examples
        except Exception as e:
            raise e

        print("Valid(Clean) Accuracy {} at epoch {}".format(accuracy_clean, epoch))
        print("Valid(Adv) Accuracy {} at epoch {}".format(accuracy_adv, epoch))





class PointAugmentTrain():
    def __init__(self, model_config, device):
        
        self._device         = device
        self._categories     = model_config['NUM_CLASSES']
        self._classifier     = PointNetCls(self._categories).to(self._device) 
        self._augmentor      = Augmentor().to(self._device)
        self._epochs            = model_config['TRAINING']['EPOCHS']
        self._optimizerA_name   = model_config['TRAINING']['OPTIMIZER']["AUGMENTOR"]
        self._optimizerC_name   = model_config['TRAINING']['OPTIMIZER']["CLASSIFIER"]
        self._lrA               = model_config['TRAINING']['LR']["AUGMENTOR"]
        self._lrC               = model_config['TRAINING']['LR']["CLASSIFIER"]
        self._pathA  = model_config['TRAINING']['PATH_AUG']
        self._pathC  = model_config['TRAINING']['PATH_CLS']

        try:
            if(self._optimizerA_name == "Adam"):
                self._optimizerA = optim.Adam(self._augmentor.parameters(),  lr  = self._lrA)
            if(self._optimizerC_name == "Adam"):
                self._optimizerC = optim.Adam(self._classifier.parameters(), lr = self._lrC)
        except:
            print("Optimizer invalid .... Proceeding with default Adam")
            self._optimizerA = optim.Adam(self._augmentor.parameters(), lr = self._lrA, eps = 1e-08,  betas = (0.9, 0.999), weight_decay = 1e-4)
            self._optimizerC = optim.Adam(self._classifier.parameters(), lr = self._lrC, eps = 1e-08, betas = (0.9, 0.999), weight_decay = 1e-4)
        else:
            print("Optimizer Augmentor  is {}".format(self._optimizerA_name))
            print("Optimizer Classifier is {}".format(self._optimizerC_name))
        
        self._lr_scheduler_classifier = torch.optim.lr_scheduler.StepLR(self._optimizerC, step_size = model_config['TRAINING']['LR_SCHEDULER']['STEP_SIZE'], 
                                        gamma = model_config['TRAINING']['LR_SCHEDULER']['GAMMA'])
        self._classifier.apply(train_utils.weights_init)
        self._augmentor.apply( train_utils.weights_init)

    def train(self, trainloader, validloader, adv = False, save_flag = True):
        if(adv):
            self.adv_train_model(trainloader, validloader)
        else:
            self.train_model(trainloader, validloader)

    def test(self, validloader, eps_list = [0.3,0.6,0.9,1.2,1.5]):
        self._classifier.load_state_dict(torch.load(self._pathC))
        self._classifier.eval()
        for eps in eps_list:
            self._eps = eps
            self.adv_valid_model(validloader)

    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, epochs):
        if epochs > 0:
            self._epochs = epochs
        else:
            print("Enter valid value of greater than zero. ")

    def save_model(self, path_aug = "./saved_augmentor.pth", path_class = "./saved_classifier.pth"):
        torch.save(self._augmentor.state_dict() , path_aug)
        torch.save(self._classifier.state_dict(), path_class)

    def train_model(self, trainloader, validloader, path = "./saved_model.pth", print_epoch = 5, save_flag = True):
        
        ispn = True
        current_best_accuracy = 0.0
        for epoch in range(self._epochs):
            self._augmentor.train()
            self._classifier.train()
            mean_correct, test_pred, test_true = [], [], []
            for idx, batch_data in enumerate(trainloader):
                self._optimizerA.zero_grad()
                self._optimizerC.zero_grad()
                
                points   = batch_data[0].permute(0,2,1).to(self._device)
                labels   = batch_data[1].to(self._device)

                # Augmentor
                self._optimizerA.zero_grad()
                aug_pc = self._augmentor(points)

                pred_pc, pc_tran, pc_feat = self._classifier(points)
                pred_aug, aug_tran, aug_feat = self._classifier(aug_pc)
                augLoss  = loss_utils.aug_loss(pred_pc, pred_aug, labels, pc_tran, aug_tran, ispn=ispn)

                augLoss.backward(retain_graph=True)
                self._optimizerA.step()


                self._optimizerC.zero_grad()
                clsLoss = loss_utils.cls_loss(pred_pc, pred_aug, labels, pc_tran, aug_tran, pc_feat,
                                              aug_feat, ispn=ispn)
                clsLoss.backward(retain_graph=True)
                self._optimizerC.step()
                
                test_true.append(labels.cpu().numpy())
                test_pred.append(torch.argmax(pred_pc, dim = 1).detach().cpu().numpy())
                pred_pc = torch.argmax(pred_pc, dim = 1)
                correct = pred_pc.eq(labels.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)

            print("Accuracy: {}".format(test_acc))
            
            self._lr_scheduler_classifier.step()


            if( ((epoch + 1) % print_epoch) == 0):
                val_accuracy = self.valid_model(validloader)
                if(val_accuracy > current_best_accuracy):
                    current_best_accuracy = val_accuracy
                    self.save_model()

                print('Valid accuracy:', val_accuracy)

        return

    def valid_model(self, validloader):
        self._classifier.eval()
        mean_correct , test_pred, test_true = [], [], []
        with torch.no_grad():
            for idx, batch_data in enumerate(validloader):
                points   = batch_data[0].permute(0,2,1).to(self._device)
                labels   = batch_data[1].to(self._device)
                pred_pc, _ , _ = self._classifier(points)
                
                test_true.append(labels.cpu().numpy())
                test_pred.append(torch.argmax(pred_pc, dim = 1).detach().cpu().numpy())
                pred_pc = torch.argmax(pred_pc, dim = 1)
                correct = pred_pc.eq(labels.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            valid_acc = metrics.accuracy_score(test_true, test_pred)
        
        return valid_acc

    def adv_valid_model(self, validloader, epoch = 0):        
        total_examples, accurate_examples, accurate_aug_examples = 0, 0, 0
        from fgsm import fgsm_attack 
        criterion = nn.CrossEntropyLoss()
        for batch_idx, batch_data in enumerate(validloader):
            #self._model.eval()
            with torch.no_grad():
                labels = batch_data[1].to(self._device)
                points = batch_data[0].permute(0,2,1).to(self._device)
                pred_pc, pc_tran, pc_feat = self._classifier(points)
                
                pred_label = torch.argmax(pred_pc, dim=1)
            
            aug_data = fgsm_attack(self._classifier, criterion, points, labels, self._eps)
            
            outputs, _ , _ = self._classifier(aug_data)
            pred_label_aug = torch.argmax(outputs, dim=1)                

            accurate_aug_examples += np.sum(pred_label_aug.cpu().numpy() == labels.cpu().numpy())
            accurate_examples += np.sum(pred_label.cpu().numpy() == labels.cpu().numpy())
            total_examples += len(pred_label.cpu().numpy())

            #visualize_batch(data, pred_label, labels, categories)
        
        try:
            accuracy_clean  = accurate_examples/total_examples
            accuracy_adv = accurate_aug_examples/total_examples
        except Exception as e:
            raise e

        print("Valid(Clean) Accuracy {} at epoch {}".format(accuracy_clean, epoch))
        print("Valid(Adv) Accuracy {} at epoch {}".format(accuracy_adv, epoch))

class Model:
    def __init__(self, opts):
        self.opts = opts
        self.backup()
        self.set_logger()

    def backup(self):
        if not self.opts.restore:
            source_folder = os.path.join(os.getcwd(),"Augment")
            common_folder = os.path.join(os.getcwd(), "Common")
            os.system("cp %s/config.py '%s/model_cls.py.backup'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/model.py '%s/model.py.backup'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/augmentor.py '%s/augmentor.py.backup'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/%s.py '%s/%s.py.backup'" % (source_folder,self.opts.model_name, self.opts.log_dir, self.opts.model_name))
            os.system("cp %s/loss_utils.py '%s/loss_utils.py.backup'" % (common_folder,self.opts.log_dir))
            os.system("cp %s/ModelNetDataLoader.py '%s/ModelNetDataLoader.py.backup'" % (common_folder,self.opts.log_dir))

    def set_logger(self):
        self.logger = logging.getLogger("CLS")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.opts.log_dir, "log_train.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def train(self):

        self.log_string('PARAMETER ...')
        self.log_string(self.opts)
        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments
        writer = SummaryWriter(logdir=self.opts.log_dir)
        
        '''DATA LOADING'''
        self.log_string('Load dataset ...')

        trainDataLoader = DataLoader(ModelNetDataLoader(self.opts, partition='train'),
                            batch_size=self.opts.batch_size, shuffle=True, drop_last=False)
        testDataLoader = DataLoader(ModelNetDataLoader(self.opts,partition='test'),
                            batch_size=self.opts.batch_size, shuffle=False)

        self.log_string("The number of training data is: %d" % len(trainDataLoader.dataset))
        self.log_string("The number of test data is: %d" % len(testDataLoader.dataset))

        '''MODEL LOADING'''
        num_class = 40
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.dim = 3 if self.opts.use_normal else 0

        classifier = PointNetCls(num_class).cuda()
        augmentor = Augmentor().cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            classifier = nn.DataParallel(classifier)
            augmentor = nn.DataParallel(augmentor)

        if self.opts.restore:
            self.log_string('Use pretrain Augment...')

            checkpoint = torch.load(self.opts.log_dir)
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('No existing Augment, starting training from scratch...')
            start_epoch = 0


        optimizer_c = torch.optim.Adam(
            classifier.parameters(),
            lr=self.opts.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.decay_rate
        )

        optimizer_a = torch.optim.Adam(
            augmentor.parameters(),
            lr=self.opts.learning_rate_a,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.decay_rate
        )
        if self.opts.no_decay:
            scheduler_c = None
        else:
            scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=20, gamma=self.opts.lr_decay)

        #scheduler_a = torch.optim.lr_scheduler.StepLR(optimizer_a, step_size=20, gamma=self.opts.lr_decay)
        scheduler_a = None

        global_epoch = 0
        best_tst_accuracy = 0.0
        blue = lambda x: '\033[94m' + x + '\033[0m'
        ispn = True if self.opts.model_name=="pointnet" else False
        '''TRANING'''
        self.logger.info('Start training...')
        PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()  # initialize augmentation

        for epoch in range(start_epoch, self.opts.epoch):
            self.log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.opts.epoch))
            if scheduler_c is not None:
                scheduler_c.step(epoch)
            if scheduler_a is not None:
                scheduler_a.step(epoch)

            for batch_id, data in enumerate(trainDataLoader, 0):
                points, target = data
                target = target[:, 0]
                points, target = points.cuda(), target.cuda().long()

                points = PointcloudScaleAndTranslate(points)
                points = points.transpose(2, 1).contiguous()

                noise = 0.02 * torch.randn(self.opts.batch_size, 1024).cuda()

                classifier = classifier.train()
                augmentor = augmentor.train()
                optimizer_a.zero_grad()
                aug_pc = augmentor(points, noise)

                pred_pc, pc_tran, pc_feat = classifier(points)
                pred_aug, aug_tran, aug_feat = classifier(aug_pc)
                augLoss  = loss_utils.aug_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, ispn=ispn)

                augLoss.backward(retain_graph=True)
                optimizer_a.step()


                optimizer_c.zero_grad()
                clsLoss = loss_utils.cls_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, pc_feat,
                                              aug_feat, ispn=ispn)
                clsLoss.backward(retain_graph=True)
                optimizer_c.step()


            train_acc = self.eval_one_epoch(classifier.eval(), trainDataLoader)
            test_acc = self.eval_one_epoch(classifier.eval(), testDataLoader)

            self.log_string('CLS Loss: %.2f'%clsLoss.data)
            self.log_string('AUG Loss: %.2f'%augLoss.data)

            self.log_string('Train Accuracy: %f' % train_acc)
            self.log_string('Test Accuracy: %f'%test_acc)
         
            writer.add_scalar("Train_Acc", train_acc, epoch)
            writer.add_scalar("Test_Acc", test_acc, epoch)


            if (test_acc >= best_tst_accuracy) and test_acc >= 0.895:# or (epoch % self.opts.epoch_per_save == 0):
                best_tst_accuracy = test_acc
                self.log_string('Save model...')
                self.save_checkpoint(
                    global_epoch + 1,
                    train_acc,
                    test_acc,
                    classifier,
                    optimizer_c,
                    str(self.opts.log_dir),
                    self.opts.model_name)

            global_epoch += 1
        self.log_string('Best Accuracy: %f' % best_tst_accuracy)
        self.log_string('End of training...')
        self.log_string(self.opts.log_dir)


    def eval_one_epoch(self, model, loader):
        mean_correct = []
        test_pred = []
        test_true = []

        for j, data in enumerate(loader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = model.eval()
            pred, _, _= classifier(points)
            pred_choice = pred.data.max(1)[1]

            test_true.append(target.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
       
        return test_acc

    def save_checkpoint(self, epoch, train_accuracy, test_accuracy, model, optimizer, path, modelnet='checkpoint'):
        savepath = path + '/%s-%f-%04d.pth' % (modelnet, test_accuracy, epoch)
        print(savepath)
        state = {
            'epoch': epoch,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            #'model_state_dict': model.module.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)

    def log_string(self, msg):
        print(msg)
        self.logger.info(msg)