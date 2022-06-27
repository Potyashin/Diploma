import torch
from torch import nn
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import scipy.stats as sps

def plot_learning_curves(history):
    '''
    Функция для вывода лосса и метрики во время обучения.

    :param history: (dict)
        accuracy и loss на обучении и валидации
    '''
    # sns.set_style(style='whitegrid')
    fig = plt.figure(figsize=(20, 14))

    plt.subplot(2,2,1)
    plt.title('Лосс', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    # plt.plot(history['loss']['val'], label='val')
    plt.ylabel('лосс', fontsize=15)
    plt.xlabel('эпоха', fontsize=15)
    plt.legend()


def weights(X_batch, interval_max):
  days = (X_batch[:, -5] - 1) * 365 + X_batch[:, -4]
  day_diff = interval_max - days + 1
  f = lambda x: x**0.5
  return f(1 / day_diff)


def train(
    model, 
    criterion,
    optimizer, 
    train_batch_gen,
    val_batch_gen,
    scheduler=None,
    num_epochs=50,
    visualize_freq=1,
    interval_max=2021 * 365 + 126,
    device='cuda'
):
    # global GLOBAL_MAX_F1_VAL
    '''
    Функция для обучения модели и вывода лосса и метрики во время обучения.

    :param model: обучаемая модель
    :param criterion: функция потерь
    :param optimizer: метод оптимизации
    :param train_batch_gen: генератор батчей для обучения
    :param val_batch_gen: генератор батчей для валидации
    :param num_epochs: количество эпох

    :return: обученная модель
    :return: (dict) accuracy и loss на обучении и валидации ("история" обучения)
    '''

    history = defaultdict(lambda: defaultdict(list))

    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        train_f1_macro = 0
        train_f1_weighted = 0
        val_loss = 0
        val_acc = 0
        val_f1_macro = 0
        val_f1_weighted = 0
        
        start_time = time.time()

        # Устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)
        # На каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in train_batch_gen: # tqdm(train_batch_gen):
            # # Аугментация
            # X_batch = transform_train(X_batch)

            # Обучаемся на батче (одна "итерация" обучения нейросети)

            # аугментация 
            mask = sps.bernoulli(p=0.7).rvs(size=torch.tensor(X_batch.shape).prod().item()).reshape(X_batch.shape)
            mask = torch.Tensor(mask)
            
            w = weights(X_batch, interval_max).to(device)

            X_batch = (X_batch * mask).to(device)
            y_batch = y_batch.to(device)

            # Логиты на выходе модели
            logits = model(X_batch)
            
            # Подсчитываем лосс
            # loss = criterion(logits.squeeze(), y_batch, w)
            loss = torch.mean(torch.square(logits.squeeze() - y_batch) * w)


            # Обратный проход
            loss.backward()
            # Шаг градиента
            optimizer.step()
            # Зануляем градиенты
            optimizer.zero_grad()
            
            # Сохраяняем лоссы и точность на трейне
            train_loss += loss.detach().cpu().numpy()

        if (scheduler is not None):
            scheduler.step()

        # Подсчитываем лоссы и сохраням в "историю"
        history['loss']['train'].append(train_loss / len(train_batch_gen))
    
        
        if (epoch % visualize_freq == 0 and epoch != 0):
          clear_output(wait=True)
          plot_learning_curves(history)
        

          # Печатаем результаты после каждой эпохи
          print("Epoch {} of {} took {:.3f}s".format(
              epoch + 1, num_epochs, time.time() - start_time))
          print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        
        
        
    return model, history
