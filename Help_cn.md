#帮助文档

##算法类型
1. RL神经网络（baseline）
2. Gecode(对照)
3. First-Fit（对照）
4. Hyper-Agent(最优算法)(first-fit + RL)

### RL baseline
0. 参数配置
参数在 config.py 文件配置。包含环境配置，神经网络配置，数据读取配置，学习/训练选择等。
1. 训练 

```python
python main.py --learn_mode=True --save_model=True --save_to=save/l12b_1 --num_epoch=20000 --min_length=12 --max_length=12 --num_layers=1 --hidden_dim=32 --num_cpus=10 --env_profile="small_default"

```
2. 测试 

```python
python main.py --learn_mode=False --save_model=False --load_model=True --load_from=save/l12b --min_length=12 --max_length=12 --num_layers=1 --hidden_dim=32 --num_cpus=10 --env_profile="small_default" --enable_performance=True
```
3.结果
  1. 训练中间结果存储位置为
```shell script
save/l12b_1/learning_history.csv
```
  2. 算法比较结果 RL神经网络 vs gecode vs first-fit(有错略过)
```shell script
save/l12b_test.csv
```

