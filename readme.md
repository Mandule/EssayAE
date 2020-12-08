# 文章自动评分任务简介

## 1 数据集简介
* 数据集的来源：Kaggle的 [Automated Essay Scoring 大赛（2012年）](https://www.kaggle.com/c/asap-aes/overview)!
* 数据集：八个文章集，各有特点。
* 文章长度：150-550英文单词。
* 标签：两个老师给出的平均分。
* 分数范围：文章集之间的分数范围不同。

| Set | Essays | Genre | AvgLen | Range | Med |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 1 | 1783 | ARG | 350 | 2-12 | 8 |
| 2 | 1800 | ARG | 350 | 1-6 | 3 |
| 3 | 1726 | RES | 150 | 0-3 | 1 |
| 4 | 1772 | RES | 150 | 0-3 | 1 |
| 5 | 1805 | RES | 150 | 0-4 | 2 |
| 6 | 1800 | RES | 150 | 0-4 | 2 |
| 7 | 1569 | RES | 150 | 0-30 | 16 |
| 8 | 723 | NAR | 650 | 0-60 | 36 |


# 2 文章题目
* Set 1 prompt: More and more people use computers, but not everyone agrees that this benefits society … Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.

* Set 2 prompt: All of us can think of a book that we hope none of our children or any other children have taken off the shelf. … Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.

* Set 3 prompt: SOURCE TEXT: Do Not Exceed Posted Speed Limit–by Joe Kurmaskie | FORGET THAT OLD SAYING ABOUT NEVER taking candy from strangers. No, a better piece of advice for the solo cyclist would be, “Never accept travel advice from a collection of old-timers who haven’t left the confines of their porches since Carter was in office … Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion.

* Set 4 prompt: SOURCE TEXT: Winter Hibiscus by Minfong Hon Saeng, a teenage girl, and her family have moved to the United States from Vietnam. As Saeng walks home after failing her driver’s test, she sees a familiar plant … Write a response that explains why the author concludes the story with this paragraph. In your response, include details and examples from the story that support your ideas.

* Set 5 prompt: SOURCE TEXT: Narciso Rodriguez, from Home: The Blueprints of Our Lives | My parents, originally from Cuba, arrived in the United States in 1956. After living for a year in … Describe the mood created by the author in the memoir. Support your answer with relevant and specific information from the memoir.

* Set 6 prompt: SOURCE TEXT: The Mooring Mast, by Marcia Amidon Lüsted | When the Empire State Building was conceived, it was planned as the world’s tallest building, taller even than the new Chrysler Building … Based on the excerpt, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.

* Set 7 prompt: Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.

* Set 8 prompt: We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.

## 3 评分指标
Quadratic Weighted Kappa (QWK): QWK是一个计算两个判断者之间的认同度。


## 4 作业要求
本次作业要求在该数据集上完成三个实验，并提交实验报告。

实验设置分为两种：

Prompt-dependent：模型在特定文章集上进行训练，并预测该文章集剩余的测试实例；

在该设置下，我们进行两个实验：

1. 基于统计方法去预测分数，需要手动设计分类特征。

2. 基于神经网络去预测分数，可使用word2vec等预训练模型；

Prompt-independent：模型在排除目标文章集上进行训练，要求预测该目标文章集上的测试实例；

在该设置下，我们进行一个实验，可以使用统计方法，也可以使用神经网络的方法。该部分实验需要思考，不同的essay set之间的文章有什么通用的信息可以利用，可以去看知识迁移的论文。

## 5 时间安排
* 第一次作业（基于统计学习的方法）

    11月29日 23:59:59
* 第二次作业（基于神经网络的方法）

    12月13日 23:59:59
* 第三次作业（prompt-independent设置）

    12月30日 23:59:59
* 实验报告

    1月12日 23:59:59
    
    
## 6 提交方式
* 线上测评
    * 测评系统网址为 47.110.235.226:12345
    * 提交结果请以 “学号.tsv” 的方式进行提交，如MG1933025.tsv
    * 提交结果应当包含8个文章集的所有预测结果
    * 提交文件中的每一行依次为Essay-ID、Essay-Set和Prediction ，利用制表符（\t）进行分割
* 实验报告
    * 第二次实验报告提交地址为：ftp://114.212.189.224
    * 截止日期为：1月15日23点59分59秒
    * 请各位同学将作业中的代码、数据及文档整理打包为zip文件。
