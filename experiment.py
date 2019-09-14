import numpy as np 
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from data import UdacityDataset, Rescale, Preprocess, ToTensor, AdvDataset
from model import BaseCNN, Nvidia, build_vgg16, Vgg16
from viewer import draw
from scipy.misc import imresize
from torchvision import datasets, transforms
from fgsm_attack import fgsm_attack
from advGAN.models import Generator
from optimization_attack import optimized_attack
import attack_test
from optimization_universal_attack import generate_noise
from advGAN_attack import advGAN_Attack
import os
from torch.utils.data import DataLoader
from scipy.misc import imsave, imread
import cv2
import os

"""
Experiment 1: test the total attack success rate of 5 attacks on 3 models 
"""
def experiment_1():
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    target_models = []
    basecnn = 'baseline.pt'
    nvidia = 'nvidia.pt'
    vgg16 = 'vgg16.pt'
    model1 = BaseCNN()
    model1.to(device)
    model1.load_state_dict(torch.load(basecnn))
    model1.eval()   
    model2 = Nvidia()
    model2.to(device)
    model2.load_state_dict(torch.load(nvidia))
    model2.eval()
    model3 = Vgg16()
    model3.to(device)
    model3.load_state_dict(torch.load(vgg16))
    model3.eval()
    target_models.append(('baseline', model1))
    # target_models.append(('vgg16', model3))
    # target_models.append(('nvidia', model2))

    root_dir = '../udacity-data'
    target = 0.3
    attacks = ('FGSM', 'Optimization', 'Optimization Universal', 'AdvGAN', 'AdvGAN Universal')
    fgsm_result = []
    opt_result = []
    optu_result = []
    advGAN_result = []
    advGANU_result = []
    fgsm_diff = []
    opt_diff = []
    optu_diff = []
    advGAN_diff = []
    advGANU_diff = []
    # models = ('baseline')

    full_indices = list(range(5614))
    test_indices = list(np.random.choice(5614, int(0.2*5614), replace=False))
    train_indices = list(set(full_indices).difference(set(test_indices)))
    image_size = (128, 128)
    # if model_name == 'baseline':
    #     image_size = (128, 128)
    # elif model_name == 'nvidia':
    #     image_size = (66, 200)
    # elif model_name == 'vgg16':
    #     image_size = (224, 224)
    test_composed = transforms.Compose([Rescale((image_size[1],image_size[0])), Preprocess(), ToTensor()])
    # train_dataset = UdacityDataset(root_dir, ['HMB1', 'HMB2', 'HMB4'], test_composed, type_='train')
    full_dataset = UdacityDataset(root_dir, ['testing'], test_composed, type_='test')

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    for (model_name, model) in target_models:

        # train_size = int(0.8*len(full_dataset))
        # test_size =len(full_dataset) - train_size

        test_data_loader = torch.utils.data.DataLoader(full_dataset,batch_size=64,shuffle=False)
        num_sample = len(full_dataset)
        # universal perturbation generation
        # if not os.path.exists(model_name + '_universal_attack_noise.npy'):
        #     print('Start universal attack training')
        #     perturbation = generate_noise(train_dataset, model, model_name, device, target)
        #     np.save(model_name + '_universal_attack_noise', perturbation)
        #     print('Finish universal attack training.')

        # # advGAN training
        # if not os.path.exists('./models/' + model_name + '_netG_epoch_60.pth'):
        # print('Start advGAN training')
        # advGAN = advGAN_Attack(model_name, model_name + '.pt', target + 0.2, train_dataset)
        # torch.save(advGAN.netG.state_dict(), './models/' + model_name +'_netG_epoch_60.pth')
        # print('Finish advGAN training')

        # # advGAN_uni training
        # if not os.path.exists('./models/' + model_name + '_universal_netG_epoch_60.pth'):
        # print('Start advGAN_uni training')
        # advGAN_uni = advGAN_Attack(model_name, model_name + '.pt', target + 0.2, train_dataset, universal=True)
        # advGAN_uni.save_noise_seed(model_name + '_noise_seed.npy')

        # torch.save(advGAN_uni.netG.state_dict(), './models/' + model_name +'_universal_netG_epoch_60.pth')
        print('Finish advGAN_uni training')
        print("Testing: " + model_name)
        #fgsm attack
        fgsm_ast, diff = fgsm_ex(test_data_loader, model, model_name, target, device, num_sample, image_size)
        print(fgsm_ast)
        fgsm_result.append(fgsm_ast)
        fgsm_diff.append(diff)
        # # optimization attack
        # opt_ast, diff = opt_ex(test_dataset, model, model_name, target, device, num_sample, image_size)
        # print(opt_ast)
        # opt_result.append(opt_ast)
        # opt_diff.append(diff)
        # optimized-based universal attack
        optu_ast, diff = opt_uni_ex(test_data_loader, model, model_name, target, device, num_sample, image_size)
        print(optu_ast)
        optu_result.append(optu_ast)
        optu_diff.append(diff)
        # advGAN attack
        advGAN_ast, diff = advGAN_ex(test_data_loader, model, model_name, target, device, num_sample, image_size)        
        print(advGAN_ast)
        advGAN_result.append(advGAN_ast)
        advGAN_diff.append(diff)
        # advGAN_universal attack
        advGANU_ast, diff = advGAN_uni_ex(test_data_loader, model, model_name, target, device, num_sample, image_size)
        print(advGANU_ast)
        # advGANU_result.append(advGANU_ast)
        # advGANU_diff.append(diff)

    # print(fgsm_result)
    # print(opt_result)
    # print(optu_result)
    # print(advGAN_result)
    # print(advGANU_result)
    # fgsm_diff = np.array(fgsm_diff)
    # np.save('fgsm_diff', fgsm_diff)
    # opt_diff = np.array(opt_diff)
    # np.save('opt_diff', opt_diff)
    # optu_diff = np.array(optu_diff)
    # np.save('optu_diff', optu_diff)
    # advGAN_diff = np.array(advGAN_diff)
    # np.save('advGAN.diff', advGAN_diff)
    # advGANU_diff = np.array(advGANU_diff)
    # np.save('advGANU.diff', advGANU_diff)
    # plt.figure(figsize=(10,5))
    # x = [0, 1.2, 2.4]
    # total_width, n = 1, 5
    # width = total_width / n
    # plt.bar(x, fgsm_result, width=width, label=attacks[0], fc = 'y')

    # for i in range(len(x)):
    #     x[i] = x[i] + width 
    # plt.bar(x, opt_result, width=width, label=attacks[1], fc = 'r')

    # for i in range(len(x)):
    #     x[i] = x[i] + width 
    # plt.bar(x, optu_result, width=width, label=attacks[2], fc = 'blue')

    # for i in range(len(x)):
    #     x[i] = x[i] + width 
    # plt.bar(x, advGAN_result, width=width, label=attacks[3], fc = 'black')
    
    # for i in range(len(x)):
    #     x[i] = x[i] + width 
    # plt.bar(x, advGANU_result, width=width, label=attacks[4], fc = 'green', tick_label=models)

    # plt.legend()
    # plt.savefig('experiment_result/experiment_1/result.jpg')

def experiment_2(gen=True):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    target_models = []
    basecnn = 'baseline.pt'
    nvidia = 'nvidia.pt'
    vgg16 = 'vgg16.pt'
    model1 = BaseCNN()
    model1.to(device)
    model1.load_state_dict(torch.load(basecnn))
    model1.eval()   
    model2 = Nvidia()
    model2.to(device)
    model2.load_state_dict(torch.load(nvidia))
    model2.eval()
    model3 = Vgg16()
    model3.to(device)
    model3.load_state_dict(torch.load(vgg16))
    model3.eval()
    target_models.append(('baseline', model1))
    target_models.append(('nvidia', model2))
    target_models.append(('vgg16', model3))

    root_dir = '../udacity-data'
    target = 0.3
    models = ('baseline', 'nvidia', 'vgg16')    
    # ex2_fun(target_models[0], target_models[1], device)
    
    attacks = ('fgsm_attack', 'opt_attack', 'universal_attack', 'advGAN_attack', 'advGAN_universal_attack')
    # blackbox test adv image  from baseline on nvidia
    print('No.1')
    result = ex2_fun(target_models[0], target_models[1], device)
    # plt_ = ex2_draw(result)
    # plt_.title('Test ' + target_models[0][0] + ' adv_image on ' +  target_models[1][0] + ' model')
    # plt_.savefig('experiment_result/experiment_2/0-1.jpg')
    # plt_.close()
    print('No.2')
    result = ex2_fun(target_models[0], target_models[2], device)
    # plt_ = ex2_draw(result)
    # plt_.title('Test ' + target_models[0][0] + ' adv_image on ' +  target_models[2][0] + ' model')
    # plt_.savefig('experiment_result/experiment_2/0-2.jpg')
    # plt_.close()
    print('No.3')

    result = ex2_fun(target_models[1], target_models[0], device)
    # plt_ = ex2_draw(result)
    # plt_.title('Test ' + target_models[1][0] + ' adv_image on ' +  target_models[0][0] + ' model')
    # plt_.savefig('experiment_result/experiment_2/1-0.jpg')
    # plt_.close()
    print('No.4')

    result = ex2_fun(target_models[1], target_models[2], device)
    # plt_ = ex2_draw(result)
    # plt_.title('Test ' + target_models[1][0] + ' adv_image on ' +  target_models[2][0] + ' model')
    # plt_.savefig('experiment_result/experiment_2/1-2.jpg')
    # plt_.close()
    print('No.5')

    result = ex2_fun(target_models[2], target_models[0], device)
    # plt_ = ex2_draw(result)
    # plt_.title('Test ' + target_models[2][0] + ' adv_image on ' +  target_models[0][0] + ' model')
    # plt_.savefig('experiment_result/experiment_2/2-0.jpg')
    # plt_.close()
    print('No.6')

    result = ex2_fun(target_models[2], target_models[1], device)
    # plt_ = ex2_draw(result)
    # plt_.title('Test ' + target_models[2][0] + ' adv_image on ' +  target_models[1][0] + ' model')
    # plt_.savefig('experiment_result/experiment_2/2-1.jpg')
    # plt_.close()
           
def experiment_3():
    pass



def ex2_draw(result):
    attacks = ('fgsm_attack', 'opt_attack', 'universal_attack', 'advGAN_attack', 'advGAN_universal_attack')
    plt.figure()
    plt.bar(range(len(result)), result, tick_label=attacks)
    return plt

def ex2_fun(gen_model, test_model, device):
    full_indices = list(range(5614))
    test_indices = list(np.random.choice(5614, int(0.2*5614), replace=False))
    root_dir = '../udacity-data'
    (gen_model_name, gen_net) = gen_model
    (test_model_name, test_net) = test_model
    image_size = (128, 128)
    # gen_image_size =None
    # if gen_model_name == 'baseline':
    #     gen_image_size = (128, 128)
    # elif gen_model_name == 'nvidia':
    #     gen_image_size = (66, 200)
    # elif gen_model_name == 'vgg16':
    #     gen_image_size = (224, 224)

    # test_image_size =None
    # if test_model_name == 'baseline':
    #     test_image_size = (128, 128)
    # elif test_model_name == 'nvidia':
    #     test_image_size = (66, 200)
    # elif test_model_name == 'vgg16':
    #     test_image_size = (224, 224)
    composed = transforms.Compose([Rescale((image_size[1],image_size[0])), Preprocess(), ToTensor()])
    # test_composed = transforms.Compose([Rescale((test_image_size[1],test_image_size[0])), Preprocess(), ToTensor()])
        # train_dataset = UdacityDataset(root_dir, ['HMB1', 'HMB2', 'HMB4'], test_composed, type_='train')
    full_dataset = UdacityDataset(root_dir, ['testing'], composed, type_='test')
    # dataset = torch.utils.data.Subset(full_dataset, test_indices)
    dataset = full_dataset
    # full_dataset = UdacityDataset(root_dir, ['testing'], test_composed, type_='test')
    # test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    # test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False)
    adv_root_path = '../udacity-data/adv_testing/'
    target = 0.3
    success = []
    attacks = ('fgsm_attack', 'opt_attack', 'universal_attack', 'advGAN_attack', 'advGAN_universal_attack')

    noise_u = np.load(gen_model_name + '_universal_attack_noise.npy')
    noise_u = torch.from_numpy(noise_u).type(torch.FloatTensor).to(device)

    advGAN_generator = Generator(3,3, gen_model_name).to(device)
    advGAN_uni_generator = Generator(3,3, gen_model_name).to(device)

    advGAN_generator.load_state_dict(torch.load('./models/' + gen_model_name + '_netG_epoch_60.pth'))    
    advGAN_uni_generator.load_state_dict(torch.load('./models/' + gen_model_name + '_universal_netG_epoch_60.pth'))    
    noise_seed = np.load(gen_model_name + '_noise_seed.npy')
    noise_a = advGAN_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
    for attack in attacks:
        total_diff = np.array([])
        adv_test_path = adv_root_path + gen_model_name + '/' + attack + '/testing/npy/'
        data_loader = iter(DataLoader(full_dataset, batch_size=64, shuffle=False))

        for i in range(88):
            adv_image = np.load(adv_test_path + 'batch_' + str(i) + '.npy')
            adv_image = torch.from_numpy(adv_image)
            adv_image = adv_image.type(torch.FloatTensor)
            adv_image = adv_image.to(device)

            ori_image = next(data_loader)['image']
            ori_image = ori_image.type(torch.FloatTensor)
            ori_image = ori_image.to(device)

            ori_y = test_net(ori_image)
            adv_y = test_net(adv_image)
            diff = (adv_y - ori_y).detach().cpu().numpy()
            diff = np.squeeze(diff)
            total_diff = np.concatenate((total_diff, diff))
        success_ = len(total_diff[abs(total_diff) >= target]) 
        print(np.mean(total_diff))
        print('test ' + gen_model_name + ' ' + attack + ' adv_image on ' +  test_model_name + ' model:', success_ / 5614)
        success.append(success_ / 5614)

    # print(len(gen_dataset))
    #for i in range(len(dataset)):
    # for i in range(88):

    #     #print(i)
    #     # gen_x = dataset[i]['image']
    #     # gen_x = gen_x.unsqueeze(0)
    #     # gen_x = gen_x.type(torch.FloatTensor)
    #     # gen_x = gen_x.to(device)
    #     # test_x = dataset[i]['image']
    #     # test_x = test_x.unsqueeze(0)
    #     # test_x = test_x.type(torch.FloatTensor)
    #     # test_x = test_x.to(device)
    #     # test_x.unsqueeze(0)     
    #     test_y_pred = test_net(test_x)
    #     # fgsm
    #     _, plt, _, perturbed_image = attack_test.fgsm_attack_test(gen_net, gen_x, target, device, image_size=image_size)
    #     # perturbed_image = perturbed_image[0,:,:,:]
    #     #imsave('experiment_result/experiment_2/' + gen_model_name + '/fgsm_attack/' + str(i+1479425441182877835) + '.jpg', perturbed_image)
    #     plt.close()
    #     # perturbed_image_resize = cv2.resize(perturbed_image, (test_image_size[1], test_image_size[0]))
    #     perturbed_image = torch.from_numpy(perturbed_image).type(torch.FloatTensor).to(device)
    #     test_y_adv = test_net(perturbed_image)
    #     # print(test_y_pred.item(), test_y_adv.item())
    #     if abs(test_y_adv.item() - test_y_pred.item()) >= target:
    #         success[0] += 1 
        
    #     _, plt, perturbed_image = attack_test.optimized_attack_test(gen_net, gen_x, target, device, image_size=image_size)
    #     #perturbed_image = perturbed_image[0,:,:,:]
    #     #imsave('experiment_result/experiment_2/' + gen_model_name + '/opt_attack/' + str(i+1479425441182877835) + '.jpg', perturbed_image)
    #     plt.close()

    #     perturbed_image = torch.from_numpy(perturbed_image).type(torch.FloatTensor).to(device)
    #     test_y_adv = test_net(perturbed_image)

    #     if abs(test_y_adv.item() - test_y_pred.item()) >= target:
    #         success[1] += 1 
        
    #     _, plt, perturbed_image = attack_test.optimized_uni_test(gen_net, gen_x, device, noise_u, image_size=image_size)
    #     #perturbed_image = perturbed_image[0,:,:,:]
    #     #imsave('experiment_result/experiment_2/' + gen_model_name + '/universal_attack/' + str(i+1479425441182877835) + '.jpg', perturbed_image)
    #     plt.close()

    #     perturbed_image = torch.from_numpy(perturbed_image).type(torch.FloatTensor).to(device)

    #     test_y_adv = test_net(perturbed_image)

    #     if abs(test_y_adv.item() - test_y_pred.item()) >= target:
    #         success[2] += 1 
        
    #     _, plt, perturbed_image = attack_test.advGAN_test(gen_net, gen_x, advGAN_generator, device, image_size=image_size)
    #     #perturbed_image = perturbed_image[0,:,:,:]
    #     #imsave('experiment_result/experiment_2/' + gen_model_name + '/advGAN_attack/' + str(i+1479425441182877835) + '.jpg', perturbed_image)
    #     plt.close()

    #     perturbed_image = torch.from_numpy(perturbed_image).type(torch.FloatTensor).to(device)

    #     test_y_adv = test_net(perturbed_image)

    #     if abs(test_y_adv.item() - test_y_pred.item()) >= target:
    #         success[3] += 1   
              
    #     _, plt, perturbed_image = attack_test.advGAN_uni_test(gen_net, gen_x, device, noise_a, image_size=image_size)
    #     #perturbed_image = perturbed_image[0,:,:,:]
    #     #imsave('experiment_result/experiment_2/' + gen_model_name + '/advGAN_universal_attack/' + str(i+1479425441182877835) + '.jpg', perturbed_image)
    #     plt.close()

    #     perturbed_image = torch.from_numpy(perturbed_image).type(torch.FloatTensor).to(device)

    #     test_y_adv = test_net(perturbed_image)

    #     if abs(test_y_adv.item() - test_y_pred.item()) >= target:
    #         success[4] += 1 
    # print('test ' + gen_model_name + ' adv_image on ' +  test_model_name + ' model:', [s/len(full_dataset) for s in success])
    return success

def fgsm_ex(test_dataset, model, model_name, target, device, num_sample, image_size=(128,128)):
    print("testing fgsm attack")
    fgsm_success = 0
    total_noise = 0
    diff_total = np.array([])
    # print(len(test_dataset))
    for i, sample in enumerate(test_dataset):
        diff, plt_, norm_noise, _ = attack_test.fgsm_attack_test(model, sample['image'], target, device, image_size=image_size)
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        # if i % 64 == 0:
        #     plt_.savefig('experiment_result/experiment_1/' + model_name + '/fgsm_attack/' + str(i) + '.jpg')
        plt_.close()  
        total_noise +=norm_noise
        fgsm_success += len(diff[diff>abs(target)])
    # print(total_noise / (1123*image_size[0]*image_size[1]*3))
    return (fgsm_success / num_sample), diff_total

def opt_ex(test_dataset, model, model_name, target, device, num_sample, image_size=(128,128)):
    print("testing optimized_based attack")
    opt_success = 0
    diff_total = np.array([])
    for i, sample in enumerate(test_dataset):
        sample['image'] = sample['image'].unsqueeze(0)
        diff, plt_, _ = attack_test.optimized_attack_test(model, sample['image'], target, device, image_size=image_size)
        diff = np.array([diff])
        diff_total = np.concatenate((diff_total, diff))
        # if i % 64 == 0:
        #     plt_.savefig('experiment_result/experiment_1/' + model_name + '/opt_attack/' + str(i/64) + '.jpg')
        plt_.close()
        if diff >= abs(target):
            opt_success += 1       
    return (opt_success / num_sample) , diff_total

def opt_uni_ex(test_dataset, model, model_name, target, device, num_sample, image_size=(128,128)):
    print("testing optimized-based universal attack")
    opt_uni_success = 0   
    diff_total = np.array([])
    for i, sample in enumerate(test_dataset):
        batch_size = sample['image'].size(0)
        noise = np.load(model_name + '_universal_attack_noise.npy')
        noise = np.tile(noise, (batch_size, 1, 1, 1))
        noise = torch.from_numpy(noise).type(torch.FloatTensor).to(device)

        diff, plt_, _ = attack_test.optimized_uni_test(model, sample['image'], device, noise, image_size=image_size)
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        # if i % 64 == 0:
        #     plt_.savefig('experiment_result/experiment_1/' + model_name + '/universal_attack/' + str(i) + '.jpg')
        plt_.close()
        # print(diff)
        opt_uni_success += len(diff[diff>abs(target)])      
    return (opt_uni_success / num_sample) , diff_total

def advGAN_ex(test_dataset, model, model_name, target, device, num_sample, image_size=(128,128)):
    print("testing advGAN attack on", model_name)
    advGAN_success = 0
    diff_total = np.array([])
    advGAN_generator = Generator(3,3, model_name).to(device)
    advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_netG_epoch_60.pth'))         
    advGAN_generator.eval() 
    for i,sample in enumerate(test_dataset):
        diff, plt_, _ = attack_test.advGAN_test(model, sample['image'], advGAN_generator, device, image_size=image_size)
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        # if i % 64 == 0:
        #     plt_.savefig('experiment_result/experiment_1/' + model_name + '/advGAN_attack/' + str(i) + '.jpg')  
        plt_.close()        
        advGAN_success += len(diff[diff >= abs(target)])
    return (advGAN_success / num_sample), diff_total

def advGAN_uni_ex(test_dataset, model, model_name, target, device, num_sample, image_size=(128,128)):
    print("testing advGAN_universal attack")
    advGAN_uni_success = 0   
    diff_total = np.array([])
    advGAN_generator = Generator(3,3, model_name).to(device)
    advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_universal_netG_epoch_60.pth'))    
    advGAN_generator.eval()
    for i, sample in enumerate(test_dataset):
        batch_size = sample['image'].size(0)
        noise_seed = np.load(model_name + '_noise_seed.npy')
        noise_seed = np.tile(noise_seed, (batch_size, 1, 1, 1))
        noise = advGAN_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
        diff, plt_, _ = attack_test.advGAN_uni_test(model, sample['image'], device, noise, image_size=image_size)
        diff = np.squeeze(diff)
        diff_total = np.concatenate((diff_total, diff))
        # if i % 64 == 0:
        #     plt_.savefig('experiment_result/experiment_1/' + model_name + '/advGAN_universal_attack/' + str(i) + '.jpg')
        plt_.close()
        advGAN_uni_success += len(diff[diff >= abs(target)])       
    return (advGAN_uni_success / num_sample), diff_total

# experiment_1()
# experiment_2()
def ex3_gen_adv(generator, gen_model, device):
    root_dir = '../udacity-data'
    adv_root_dir = '../udacity-data/adv_data'
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    image_size = (128, 128)
    test_composed = transforms.Compose([Rescale(image_size), Preprocess(), ToTensor()])
    basecnn = 'baseline.pt'
    nvidia = 'nvidia.pt'
    vgg16 = 'vgg16.pt'
    model1 = BaseCNN()
    model1.to(device)
    model1.load_state_dict(torch.load(basecnn))
    model1.eval() 
    model2 = Nvidia()
    model2.to(device)
    model2.load_state_dict(torch.load(nvidia))
    model2.eval()
    model3 = Vgg16()
    model3.to(device)
    model3.load_state_dict(torch.load(vgg16))
    model3.eval()
    target_models = []
    target_models.append(('baseline', model1))
    #target_models.append(('nvidia', model2))
    target_models.append(('vgg16', model3))
    train = 0    
    attacks = ['advGAN_attack']
    # attacks = ['fgsm_attack', 'universal_attack', 'advGAN_attack', 'advGAN_universal_attack', 'opt_attack',]
    target = 0.3
    if train:
        hmb_list =[('HMB1', 1479424215880976321),('HMB2', 1479424439139199216),('HMB4', 1479425729831388501), ('HMB5', 1479425834048269765), ('HMB6', 1479426202229245710)]
    else:
        hmb_list = [('testing', 1479425441182877835)]
    # for (model_name, model) in target_models:
    #     noise_u = np.load(model_name + '_universal_attack_noise.npy')
    #     noise_u = torch.from_numpy(noise_u).type(torch.FloatTensor).to(device)

    #     advGAN_generator = Generator(3,3, model_name).to(device)
    #     advGAN_uni_generator = Generator(3,3, model_name).to(device)

    #     advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_netG_epoch_60.pth'))    
    #     advGAN_uni_generator.load_state_dict(torch.load('./models/' + model_name + '_universal_netG_epoch_60.pth'))    
    #     noise_seed = np.load(model_name + '_noise_seed.npy')
    #     noise_a = advGAN_uni_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
    #     save_dir = os.path.join(adv_root_dir, model_name)
    for (model_name, model) in target_models:
        noise_u = np.load(model_name + '_universal_attack_noise.npy')
        noise_u = torch.from_numpy(noise_u).type(torch.FloatTensor).to(device)

        advGAN_generator = Generator(3,3, model_name).to(device)
        advGAN_uni_generator = Generator(3,3, model_name).to(device)

        advGAN_generator.load_state_dict(torch.load('./models/' + model_name + '_netG_epoch_60.pth'))    
        advGAN_generator.eval()
        advGAN_uni_generator.load_state_dict(torch.load('./models/' + model_name + '_universal_netG_epoch_60.pth'))    
        advGAN_uni_generator.eval()
        noise_seed = np.load(model_name + '_noise_seed.npy')
        noise_a = advGAN_uni_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
        save_dir = os.path.join(adv_root_dir, model_name)
        for (hmb, start) in hmb_list:
            print(model_name ,hmb)
            if train:
                train_dataset = UdacityDataset(root_dir, [hmb], test_composed, type_='train')
            else:
                train_dataset = UdacityDataset(root_dir, [hmb], test_composed, type_='test')
            generator = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)
            for i, batch in enumerate(generator):
                batch_x = batch['image']
                batch_x = batch_x.type(torch.FloatTensor)
                batch_x = batch_x.to(device)

                _, plt, _, perturbed_image = attack_test.fgsm_attack_test(model, batch_x, target, device, image_size=image_size)
                plt.close()
                if train:
                    for j in range(len(perturbed_image)):
                        np.save('../udacity-data/adv_data/' + model_name + '/fgsm_attack/' + hmb + '/' + str(i*64 + start + j), perturbed_image[j,:,:,:])
                else:
                    np.save('../udacity-data/adv_testing/' + model_name + '/fgsm_attack/' + hmb + '/npy/' + 'batch_' + str(i), perturbed_image)

                _, plt, perturbed_image = attack_test.optimized_uni_test(model, batch_x, device, noise_u, image_size=image_size)
                plt.close()

                if train:
                    for j in range(len(perturbed_image)):
                        np.save('../udacity-data/adv_data/' + model_name + '/universal_attack/' + hmb + '/' + str(i*64 + start + j), perturbed_image[j,:,:,:])
                else:
                    np.save('../udacity-data/adv_testing/' + model_name + '/universal_attack/' + hmb + '/npy/' + 'batch_' + str(i), perturbed_image)
        
                _, plt, perturbed_image = attack_test.advGAN_test(model, batch_x, advGAN_generator, device, image_size=image_size)
                plt.close()
                
                if train:
                    for j in range(len(perturbed_image)):
                        np.save('../udacity-data/adv_data/' + model_name + '/advGAN_attack/' + hmb + '/' + str(i*64 + start + j), perturbed_image[j,:,:,:])
                else:
                    np.save('../udacity-data/adv_testing/' + model_name + '/advGAN_attack/' + hmb + '/npy/' + 'batch_' + str(i), perturbed_image)
                
                _, plt, perturbed_image = attack_test.advGAN_uni_test(model, batch_x, device, noise_a, image_size=image_size)
                plt.close()

                if train:
                    for j in range(len(perturbed_image)):
                        np.save('../udacity-data/adv_data/' + model_name + '/advGAN_universal_attack/' + hmb + '/' + str(i*64 + start + j), perturbed_image[j,:,:,:])
                else:
                    np.save('../udacity-data/adv_testing/' + model_name + '/advGAN_universal_attack/' + hmb + '/npy/' + 'batch_' + str(i), perturbed_image)
    for (model_name, model) in target_models:
        for (hmb, start) in hmb_list:
            print(model_name, hmb)
            if train:
                train_dataset = UdacityDataset(root_dir, [hmb], test_composed, type_='train')
            else:
                train_dataset = UdacityDataset(root_dir, [hmb], test_composed, type_='test')
            # npy = np.array([], dtype=np.float64).reshape(1, 3, 128, 128)
            npy = None

            for i in range(0, len(train_dataset)):

                batch_x = train_dataset[i]['image']
                batch_x = batch_x.unsqueeze(0)
                batch_x = batch_x.type(torch.FloatTensor)
                batch_x = batch_x.to(device)
                _, plt, perturbed_image = attack_test.optimized_attack_test(model, batch_x, target, device, image_size=image_size)
                plt.close()
                if train:
                    for j in range(len(perturbed_image)):
                        np.save('../udacity-data/adv_data/' + model_name + '/opt_attack/' + hmb + '/' + str(i*64 + start + j), perturbed_image[j,:,:,:])
                else:
                    if i == 0:
                        npy = perturbed_image
                    elif i % 64 != 0:
                        npy = np.concatenate((npy, perturbed_image))
                    else:
                        np.save('../udacity-data/adv_testing/' + model_name + '/opt_attack/' + hmb + '/npy/' + 'batch_' + str(i // 64 - 1), npy)
                        npy = perturbed_image
            

            if not train:
                np.save('../udacity-data/adv_testing/' + model_name + '/opt_attack/' + hmb + '/npy/' + 'batch_' + str(5614 // 64), npy)

if __name__ == "__main__":
    # experiment_1()
    experiment_2()


                
    
