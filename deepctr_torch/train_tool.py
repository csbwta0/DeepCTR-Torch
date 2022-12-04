import torch
import time,tqdm
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler,DataLoader,TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

# Deep-CTR
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers.utils import slice_arrays


# 一些关于DDP优化的trick
# 1. 模型中的module创建模块最好和网络计算顺序保持一致
# 2. 模型的optimizer最好在DDP模型创建后再初始化
# 3. 模型的参数在DDP模型创建之后不应当再调整

# TODO：检查bug   
# DDP模型和单卡模型实验的性能应该相近, 应该有bug
# 1. 检查数据同步问题
# 2. 针对不同进程采用不同seed？
# 3. distributedsampler作用
def parallel_train_SNMG(local_rank,world_size,model:BaseModel,x,y,epochs,batch_size=None,optimizer_type=None,lr=1e-4,
                        loss_type=None,metrics:list=None,validation_split=0,validation_data=None,checkpoint_interval=0):
    """单节点多卡并行训练
    
    Args:
        - local_rank: 进程序号 
        - world_size: 进程总数
        - model: 
        - x: 
        - y:
        - epochs:
        - batch_size: 
        - optimizer_type: 优化器类型 
        - lr: 学习率
        - loss_type: 
        - metrics:
        - validation_split:
        - valdation_data:
        - checkpoint_interval: 保存模型间隔, default=0 不保存模型
    """
    
    dist.init_process_group(backend='gloo',rank=local_rank,world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    if isinstance(x,dict):
        x = ([x[feat] for feat in model.feature_index])
    
    # Validation Data Prepare
    
    val_x = []
    val_y = []
    
    do_validation = False
    # 验证集数据
    if validation_data:
        do_validation = True
        if len(validation_data) == 2:
            val_x, val_y = validation_data
            val_sample_weight = None
        
        elif len(validation_data) == 3:
            val_x,val_y,val_sample_weight = validation_data
        
        else:
            raise  ValueError(
                'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
        
        if isinstance(val_x,dict):
            val_x = [val_x[feat] for feat in model.feature_index]
    
    # 验证集比例划分
    elif 0.< validation_split < 1.0:
        do_validation = True
        if hasattr(x[0],'shape'):
            split_at = int(x[0].shape[0]*(1-validation_split))
        else:
            split_at = int(len(x[0])*(1-validation_split))
            
        x,val_x = slice_arrays(x,0,split_at), slice_arrays(x,split_at)
        y,val_y = slice_arrays(y,0,split_at), slice_arrays(y,split_at)
        
    # 验证集为空
    else:
        val_x = []
        val_y = []
    
    
    # Train Data Prepare
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i],axis=1)
            
    train_tensor_data = TensorDataset(
        torch.from_numpy(np.concatenate(x,axis=-1)),
        torch.from_numpy(y)
    )
    
    if batch_size is None:
        batch_size = 256
    
    train_sampler = DistributedSampler(train_tensor_data,num_replicas=world_size,rank=local_rank)
    
    train_dataloader = DataLoader(train_tensor_data,batch_size=batch_size,sampler=train_sampler)
    
    
    
    # DDP Prepare
    model = model.cuda()
    model.device = 'cuda:{}'.format(local_rank)
    model.linear_model.device = 'cuda:{}'.format(local_rank)
    
    model = DDP(module=model,device_ids=[local_rank],output_device=local_rank)
    
    model.module.compile(optimizer_type,lr=lr,loss=loss_type,metrics=metrics)
    
    
    torch.manual_seed(local_rank)
    torch.cuda.manual_seed(local_rank)
    # 要在DDP model创建之后初始化optimizer
    # Set Optimizer
    optimizer = model.module.optim
    
    criterion = model.module.loss_func
    # Train
    sample_num = len(train_tensor_data)
    steps_per_epoch = (sample_num-1)//batch_size+1
    print("Local Rank:{0}, Train on {1} samples, validate on {2} samples, {3} steps per epoch".format(
        local_rank,sample_num,len(val_y),steps_per_epoch
    )) 
    
    
    for epoch in range(epochs):
        start_time = time.time()
        loss_epoch = 0.0
        total_loss_epoch = 0
        train_result = {}
        epoch_logs = {}
        # train_dataloader.sampler.set_epoch(epoch)
        try:
            for train_x,train_y in train_dataloader:
                x = train_x.float().cuda()
                y = train_y.float().cuda()
                
                pred_y = model(x).squeeze()
                
                optimizer.zero_grad()
                
                # multi task learning
                if isinstance(criterion,list):
                    assert len(criterion) == model.module.num_tasks, "the length of `loss_func` should be equal `self.num_tasks`"
                    loss = sum([criterion[i](pred_y[:,i],y[:,i],reduction='sum') for i in range(model.module.num_tasks)])
                else:
                    loss = criterion(pred_y,y.squeeze(),reduction='sum')
                
                reg_loss = model.module.get_regularization_loss()
                
                total_loss = loss + reg_loss + model.module.aux_loss.cuda()
                
                loss_epoch += loss.item()
                total_loss_epoch += total_loss.item()
                
                total_loss.backward()
                optimizer.step()
                
                # 记录epoch训练结果
                for name,metric_fun in model.module.metrics.items():
                    if name not in train_result:
                        train_result[name] = []
                    train_result[name].append(metric_fun(
                        y.cpu().data.numpy(),pred_y.cpu().data.numpy().astype('float64')))
                        
        except KeyboardInterrupt:
            print("Close")
            raise 
        
        # TODO: 根据进程号过滤输出信息
        # 可以用logging, 根据level控制输出和文件写入
        epoch_logs['loss'] = total_loss_epoch/sample_num
        for name,result in train_result.items():
            epoch_logs[name] = np.sum(result)/steps_per_epoch
        
        
        if do_validation and local_rank == 0:
            eval_result = model.module.evaluate(val_x,val_y,batch_size)
            for name,result in eval_result.items():
                epoch_logs["val_"+name] = result
        
        
        epoch_time = (time.time()-start_time)
        print("Local Rank:{0} Epoch {1}/{2}".format(local_rank,epoch+1,epochs))
        
        eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

        for name in metrics:
            eval_str += " - " + name + \
                        ": {0: .4f}".format(epoch_logs[name])
        
        if do_validation and local_rank == 0:
            for name in metrics:
                eval_str += " - "+ "val_" + name +": {0: .4f}".format(epoch_logs["val_"+name]) 
            
        print(eval_str)
        
        
        if local_rank == 0 and (epoch+1)%checkpoint_interval == 0:
            torch.save(model.module.state_dict(),"./model.pth")
    
    
def parallel_train_MNMG():
    """多节点多卡并行训练
    
    """
    pass
    