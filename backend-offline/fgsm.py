import torch
import torch.nn as nn

class Attack:
    def __init__(self, model, epsilon=0.25):
        self.model = model
        self.epsilon = epsilon

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
    
    def generate_adversarial(self, data, target, criterion):
        data.requires_grad = True
        output = self.model(data)
        loss = criterion(output, target)
        self.model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = self.fgsm_attack(data, self.epsilon, data_grad)
        return perturbed_data