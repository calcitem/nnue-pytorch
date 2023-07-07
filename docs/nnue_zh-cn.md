# NNUE

## 前言

本文档包含的内容：

- 技术含量
- NNUE及其原理的详细描述
- 快速线性代数复习
- 输入定义和分解
- 适用于 NNUE 网络的组件（层）
- 推理代码和优化
- 量化数学和实现
- 几乎生产就绪的优化代码
- pytorch 训练器实现（+ 重要的 CUDA 内核）
- 架构考虑和历史

本文档不包含的内容：

- 训练网络的教程（参见 [the wiki](https://github.com/glinscott/nnue-pytorch/wiki)）
- 数据集、优化器、超参数
- 实验结果日志

＃＃ 目录

* [前言](#前言)
* [目录](#目录)
* [基础知识](#basics)
     + [什么是 NNUE？](#what-is-nnue)
         - [量化 101 及其重要性](#quantization-101-and-its-importance)
     + [NNUE 中哪些层有用？](#what-layers-are-useful-in-nnue)
         - [线性层](#linear-layer)
         - [具有稀疏输入的线性层](#linear-layer-with-sparse-inputs)
         - [裁剪 ReLU 层](#clipped-relu-layer)
         - [乙状结肠](#乙状结肠)
         - [Quantmoid4](#quantmoid4)
         - [池化层](#pooling-layers)
     + [一个简单的输入特征集。](#a-simple-input-feature-set)
     + [一个简单的 NNUE 网络](#a-simple-nnue-network)
     + [考虑网络规模和成本。](#consideration-of-networks-size-and-cost)
         - [功能集](#feature-set)
         - [第一组隐藏神经元](#first-set-of-hidden-neurons)
         - [更多层](#further-layers)
     + [累加器](#累加器)
     + [半KP](#halfkp)
         - [多视角，多累加器](#multiple-perspectives--multiple-accumulators)
             * [如何组合多个累加器视角？](#how-to-combine-multiple-accumulator-perspectives)
             * [每个视角使用哪一组权重？](#which-set-of-weights-to-use-for-each-perspective)
         - [HalfKP 示例和网络图](#halfkp-example-and-network-diagram)
* [前向传递实现](#forward-pass-implementation)
     + [示例网络](#example-network)
     + [图层参数](#layer-parameters)
     + [累加器](#accumulator-1)
         - [刷新累加器](#refreshing-the-accumulator)
         - [更新累加器](#updating-the-accumulator)
     + [线性层](#linear-layer-1)
     + [ClippedReLU](#clippedrelu)
     + [把它放在一起](#putting-it-together)
* [用 pytorch 训练网络](#training-a-net-with-pytorch)
     + [模型规格](#model-specification)
     + [准备输入](#preparing-the-inputs)
         - [解析训练数据集并将它们移动到 python 端](#parsing-the-training-data-sets-and-moving-them-to-the-python-side)
         - [训练批结构和通信](#training-batch-structure-and-communication)
     + [特征分解](#feature-factorization)
         - [虚拟特征合并](#virtual-feature-coalescing)
         - [其他因素](#other-factors)
             * [“K”因子](#k-因子)
             * [“HalfRelativeKP”因素](#halfrelativekp-factors)
         - [因式分解器的真实效果](#real-effect-of-the-factorizer)
     + [损失函数及其应用方法](#loss-functions-and-how-to-apply-them)
         - [目标](#the-goal)
         - [将评估从 CP 空间转换为 WDL 空间](#converting-the-evaluation-from-cp-space-to-wdl-space)
         - [在评估中使用结果](#using-results-along-the-evaluation)
         - [均方误差 (MSE)](#mean-squared-error-mse)
             * [损失](#损失)
             * [毕业](#毕业)
         - [交叉熵](#cross-entropy)
             * [损失](#loss-1)
             * [毕业](#grad-1)
* [量化](#量化)
     + [Stockfish 量化方案](#stockfish-quantization-scheme)
         - [特征转换器](#feature-transformer)
         - [线性层](#linear-layer-2)
         - [ClippedReLU](#clippedrelu-1)
     + [量化数学以及如何使其适合](#the-math-of-quantization-and-how-to-make-it-fit)
         - [特征转换器](#feature-transformer-1)
         - [线性层](#linear-layer-3)
     + [实施](#实施)
     + [优化实施](#optimized-implementation)
         - [特征转换器](#feature-transformer-2)
         - [线性层](#linear-layer-4)
             * [m256_add_dpbusd_epi32](#m256_add_dpbusd_epi32)
             * [m256_haddx4](#m256_haddx4)
         - [具有稀疏输入的线性层](#linear-layer-with-sparse-input)
             * [m256_process_chunk](#m256_process_chunk)
         - [具有稀疏输入的线性层，替代方法](#linear-layer-with-sparse-input-alternative-approach)
             * [m256_process_chunk_alternative](#m256_process_chunk_alternative)
         - [具有稀疏输入和阻塞稀疏输出的线性层](#linear-layer-with-sparse-input-and-blocked-sparse-output)
         - [ClippedReLU](#clippedrelu-2)
             * [int16 -> int8](#int16---int8)
             * [int32 -> int8](#int32---int8)
         - [Quantmoid4](#quantmoid4-1)
         - [池化层](#pooling-layers-1)
             * [平均池化](#average-pooling)
             * [最大池化](#max-pooling)
             * [产品池化](#product-pooling)
     + [在训练器中计算量化](#accounting-for-quantization-in-the-trainer)
         - [范围](#range)
             * [优化器内部](#inside-the-optimizer)
             * [在优化器之外](#outside-the-optimizer)
             * [计算虚拟层（分解）](#accounting-for-virtual-layers-factorization)
         - [不可微分层](#non-differentiable-layers)
             * [用于训练安全合并 Quantmoid4 的自定义内核](#custom-kernel-for-training-safe-amalgamated-quantmoid4)
* [优化培训师 (CUDA)](#optimizing-the-trainer-cuda)
     + [使用自定义 CUDA 内核](#using-custom-cuda-kernels)
     + [特征转换器](#feature-transformer-3)
         - [新数据加载器](#new-data-loader)
         - [特征转换器前向内核](#feature-transformer-forward-kernel)
         - [特征转换器后向内核](#feature-transformer-backward-kernel)
         - [FeatureTransformerSlice 层](#featuretransformerslice-layer)
* [架构和新方向](#architectures-and-new-directions)
     + [简单的 HalfKP Stockfish 架构](#simple-halfkp-stockfish-architecture)
     + [HalfKAv2 功能集。](#halfkav2-feature-set)
     + [HalfKAv2_hm 特征集。](#halfkav2_hm-feature-set)
     + [特征转换器的一部分直接转发到输出。](#a-part-of-the-feature-transformer-directly-forwarded-to-the-output)
     + [多个 PSQT 输出和多个子网](#multiple-psqt-outputs-and-multiple-subnetworks)
* [历史 Stockfish 评估网络架构](#historical-stockfish-evaluation-network-architectures)
     + [“SFNNv5”架构](#sfnnv5-架构)
     + [“SFNNv4”架构](#sfnnv4-架构)
     + [“SFNNv3”架构](#sfnnv3-架构)
     + [“SFNNv2”架构](#sfnnv2-架构)
     + [“SFNNv1”架构](#sfnnv1-架构)
	

## 基本

### 什么是NNUE？

从广义上讲，NNUE（ƎUИИ 高效可更新神经网络）是一种神经网络架构，它利用了后续评估之间网络输入变化最小的优势。 它是由 ![Yu Nasu](https://www.chessprogramming.org/Yu_Nasu) 为将棋发明的，集成到 Motohiro Isozaki 于 2018 年 5 月开发的 [YaneuraOu](https://github.com/yaneurao/YaneuraOu) ，后来由 ![Hisayori Noda](https://www.chessprogramming.org/Hisayori_Noda) 于 2019 年 6 月移植到国际象棋中用于 Stockfish，但它适用于许多其他棋盘游戏，甚至可能适用于其他领域。 NNUE 的运作遵循以下原则：

1. 网络应该有相对较少的非零输入。
2. 输入在后续评估之间应尽可能少地改变。
3. 网络应该足够简单，便于整数域的低精度推理。

遵循第一个原则意味着当网络规模扩大时，输入必须变得稀疏。 当前最佳架构的输入稀疏度约为 0.1%。 在必须对网络进行整体评估的情况下，少量非零输入会降低评估网络所需时间的上限。 这就是为什么 NNUE 网络可以很大，同时仍然可以非常快速地评估的主要原因。

遵循第二个原则（前提是遵循第一个原则）创建了一种有效更新网络（或至少其中昂贵的部分）而不是重新评估整个网络的方法。 这利用了这样一个事实，即一次移动只会稍微改变棋盘状态。 这比第一个原则的重要性要低，并且对于要利用的实现来说是完全可选的，但是尽管如此，它还是会在确实注意利用它的实现中提供可衡量的改进。

遵循第三条原则可以在通用硬件上实现最高性能，并使该模型特别适合传统国际象棋引擎所必需的低延迟 CPU 推理。

总体而言，NNUE 原则也适用于昂贵的深度网络，但它们在快速浅层网络中大放异彩，适用于低延迟 CPU 推理，无需批处理和加速器。 目标性能是每个线程每秒进行百万次评估。 这是一个极端的用例，需要极端的解决方案，最重要的是**量化**。

####量化101及其重要性

量化是将神经网络模型的域从浮点数变为整数的过程。 NNUE 网络旨在在低精度整数域中进行快速评估，并可以最大程度地利用现代 CPU 可用的 int8/int16 性能。 浮点不是实现最大引擎强度的选项，因为它牺牲了太多的速度以获得太少的精度增益（尽管一些引擎由于其简单性而使用浮点表示）。 量化不可避免地会引入误差，网络越深，误差越多，但是对于相对较浅的 NNUE 网络，这个误差可以忽略不计。 稍后将在本文档中更详细地描述量化。 到那时本文档将使用浮点数而不是整数，在我们进行实际代码优化之前它并不重要。 这种感叹的目的是让读者了解 NNUE 的最终目标，因为它是塑造 NNUE 模型并决定什么是可能的和什么不是的最大因素。

### NNUE 中哪些层有用？

NNUE 依赖于可以使用简单算法在低精度环境中实现的简单层。 这意味着 Linear（完全连接，基本上是矩阵乘法）和 ClippedReLU (clamp(0, 1)) 层特别适合它。 池化层 (mul/avg/max) 或更复杂的激活函数（如 sigmoid）的近似值也适用但不常用。

通常这样的网络保持浅层（2-4 层），因为大部分知识都保存在第一层（利用输入稀疏性来保持性能）并且在第一层之后网络需要急剧减少其宽度（好处是 网络后面部分的更深部分将受到大的第一层的影响）以保持性能要求。

#### 线性层

线性（全连接）层只是一个简单的矩阵乘法。 它可以高效地实现，支持稀疏输入，并提供良好的容量。 它将“in_features”值作为输入，并生成“out_features”值。 操作是“y = Ax+b”，其中：

`x` - 大小为 `in_features` 的输入列向量

`A` - 大小为 `(out_features, in_features)` 的权重矩阵

`b` - 大小为 `out_features` 的偏置列向量

`y` - 大小为 `out_features` 的输出列向量

![矩阵向量乘法](img/mv.svg)

#### 具有稀疏输入的线性层

乘法 Ax 在概念上可以简化为“如果 x[i] 不为零，则从 A 中取出 i 列，将其乘以 x[i] 并将其添加到结果中”。 现在应该很明显，只要输入的一个元素为零，我们就可以跳过处理权重矩阵的整行。 这意味着我们只需要处理与输入向量中的非零值一样多的“A”列。 尽管权重矩阵中可能有数万列，但我们只关心每个位置的其中几列！ 这就是为什么第一层可以这么大。

![矩阵和稀疏向量乘法](img/mvs.svg)

#### 裁剪的 ReLU 层

这是一个基于普通 ReLU 的激活函数，不同之处在于它从下方和上方都有界。 公式为“y = min(max(x, 0), 1)”。

![ClippedReLU](img/clipped_relu.png)

该层的目的是为网络添加非线性。 如果它只是线性层，它们可以全部折叠成一个，因为矩阵可以相乘在一起。

ClippedReLU 最好用 ReLU 代替，但激进的量化需要减少隐藏层输入的动态范围，因此将值上限设为 1 对性能很重要。

#### 乙状结肠

这是一个激活函数，与 [clipped] ReLU 相反，它是平滑的。 公式为“y = 1/(1+e^-kx)”，其中 k 是确定形状“拉伸”程度的参数。

![Sigmoid](img/sigmoid.png)

与 clipped ReLU 相比有两个主要区别：

1. sigmoid是光滑的，意思是处处可微，意思是不存在（现实地说）梯度消失的情况。
2. sigmoid 是非线性的，输出向 0 或 1 饱和但永远不会达到

虽然此功能通常允许网络比 ReLU 学习更多，但它成本高昂且不适合在整数域中进行评估。 然而，这是一个很好的改进起点......

#### Quantmoid4

由于 sigmoid 的成本太高，我们需要寻找替代方案。 一种这样的替代方法是使用近似值。 碰巧的是，`sigmoid(4x)`（以特定方式缩放到整数域）可以通过一个简单的分段二次函数很好地近似，该函数只需要加法、乘法和位移。 由于这种近似的主要目的是直接用于量化实现，我们将提供一个特定的变体，输出范围为“[0, 126]”的值（并相应地缩放输入）。 选择上限定义为 126 的原因是这是最大的偶数 8 位整数，我们想要一个偶数以允许 `x=0` 的值恰好位于中间。 等式如下：

![Quantmoid4 方程](img/quantmoid4_equation.png)

请注意，正负“x”的方程式几乎相同。 即使有两种情况，相似性也允许无分支实现。

结果图如下（带有缩放的 sigmoid(4x) 进行比较）：

![Quantmoid4](img/quantmoid4.png)

缺点是它失去了平滑度，并且输出很早就舍入到 0/1。 然而，这在实践中似乎不是问题，这种“四舍五入”的实际误差可以忽略不计。

一旦我们实施和优化它，就会发生更多很棒的事情，所以我们将回到优化量化实施部分的这一层。

#### 池化层

有时需要降低输入维度以使层的大小更易于理解。 例如，与其使用输出非常窄的“1024->8”层，不如使用“512->16”。 池化层可以通过降低维度来提供一些灵活性。

池化层通过在输入的非重叠跨度上应用函数“F”来工作，其中“F”的输入多于输出。 因此，例如，可以让“F”接受 2 个连续输入并产生一个输出，从而有效地将神经元数量减半。

可以考虑以下类型的池化层：

1. Average Pooling——输出输入的平均值。 适用于任意数量的输入。
2. Max Pooling - 输出输入的最大值。 适用于任意数量的输入。
3. Product Pooling——输出输入的乘积。 由 Stockfish 引入，一般在机器学习中并不常见。 仅适用于 2 个输入。 这个似乎也有与 sigmoid (quantmoid4) 相似的好处； 它增加了网络的容量，而其他池化层只允许减少维度。

### 一个简单的输入特征集。

出于说明的目的，我们将考虑一组基于棋子放置的简单输入。 我们将其称为“A”特征，因为它们将代表“所有部分”。

棋盘上有 64 个方格，6 种棋子类型（兵、马、象、车、王后、国王）和 2 种颜色（白、黑）。 我们要编码为输入的是棋子的位置，因此每个输入都对应于一些（正方形、棋子类型、颜色）元组。 有 64*6*2=768 个这样的元组。 如果在正方形 `S` 上有一块颜色为 `C` 的 `P`，我们将输入 `(S, P, C)` 设置为 1，否则我们将其设置为 0。即使输入的总数是 768 在任何给定的合法棋局中只能有 32 个非零输入，因为棋盘上最多只有 32 个棋子。 而且，任何一步最多只能改变 4 个输入（castling），平均应该在 3 个以下。

在将特征传递给神经网络时，利用了输入的二进制和稀疏性质——输入只是特征列表（索引），不需要完整的输入向量，因为其他位置的值为 0，我们知道每个 活动功能具有与之关联的值 1。

让我们看一个示例位置 `1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1`。

![](img/board_0.png)

在上面的棋盘上，我们有 4 个活动特征：`(A1, king, white)`, `(C3, pawn, white)`, `(B8, king, black)`, `(D4, rook, black)`。

现在让我们考虑移动 c4 - 唯一无效的特征是 `(C3, pawn, white)`，它需要替换为 `(C4, pawn, white)`。

现在让我们考虑移动 cxd4 - 棋子移动了，就像我们删除“(C3, pawn, white)”并添加“(D4, pawn, white)”之前一样。 但是车也从棋盘上移除了，所以我们也必须移除`(D4, rook, black)`。 这仍然比从头开始重新创建输入要少工作！

### 一个简单的NNUE网络

我们将使用上一段中的“A”特征集，因此我们有 768 个输入。 用于此说明的层将是 3 个线性层，768->8、8->8、8->1。 所有层都是线性的，所有隐藏神经元都使用 ClippedReLU 激活函数。 下图说明了架构：

![A[768]->8->8->1架构图](img/A-768-8-8-1.svg)

流动是从左到右。 第一层是一个大型全连接层，有 768 个输入，但每个位置只有一小部分是非零的 - 可以使用稀疏矩阵向量乘法。 隐藏层要小得多，并且总是用密集矩阵向量乘法计算。 最后我们得到 1 个输出，它通常被训练为位置的 centipawn 评估（或与其成比例）。

### 考虑网络规模和成本。

选择正确的架构很棘手，因为它是准确性/性能的权衡。 大型网络提供更准确的评估，但速度影响可能会完全抵消实际游戏中的收益。 例如，Stockfish 从“256x2->32->32->1”缓慢过渡到“1024x2->8->32->1”。

#### 功能集

在选择功能集时，可能很想了解复杂的领域特定知识，但相关的成本使更简单的解决方案更具吸引力。 HalfKP，后面会详细解释，非常简单，速度快，还不错。 已经尝试了更复杂的功能集，但它们通常无法解决性能问题。 HalfKP 特征易于计算，并且位置与位置之间的变化很小。

大小也必须考虑。 对于“256x2->32->32->1”架构，HalfKP 输入在第一层需要大约 1000 万个参数，量化后相当于 20MB。 对于某些用途，拥有非常大的特征集可能不是问题，可能有数亿个参数，但对于典型用户而言，这会带来不便。 此外，增加特征集大小可能会降低某些实现的训练速度，并且肯定需要更多时间才能收敛。

#### 第一组隐藏神经元

第一层的输出数量是最关键的参数，对速度和尺寸的影响也最大。 与此参数相关的成本是双重的。 一方面，它增加了更新累加器时所需的操作次数。 其次，对于优化的实现，必须考虑可用寄存器的数量——在超过 256 个神经元的 Stockfish 中需要多次传递特征索引，因为 AVX2 没有足够的寄存器。 它还部分决定了第一个密集线性层的大小，这也对总成本有很大贡献。

#### 更多层

与机器学习中考虑的典型网络不同，这里的大部分知识都存储在第一层，因此在输出附近添加更多的小层对准确性几乎没有影响，如果由于误差累积而采用量化，甚至可能有害。 NNUE 网络保持异常浅，保持后面层的尺寸小可以提高性能。

### 累加器

尽管我们观察到很少有输入随着位置的变化而变化，但我们还没有利用它。 回想一下，线性层只是将一些权重矩阵列加在一起。 我们可以将它们保留为位置状态的一部分，而不是为每个位置重新计算第一组隐藏神经元，并根据添加或删除的特征（列）在每次移动时更新它！ 我们只需要处理两种简单的情况：

1. 从输入 (1 -> 0) 中删除了特征 `i` - 从累加器中减去权重矩阵的列 `i`
2. 将特征“i”添加到输入 (0 -> 1) - 将权重矩阵的列“i”添加到累加器

对于一个单一的移动，找到哪些“A”特征发生了变化是微不足道的——我们知道我们正在移动什么部分，从哪里，到哪里。 捕获和提升可以被视为一块消失或无处出现的棋子。

但是，在使用浮点值时必须小心。 重复添加和减去浮点数会导致错误随着每次移动而累积。 需要仔细评估误差是否小到足以让网络仍然产生良好的结果。 值得庆幸的是，最好的实现方式是在撤消移动时不更新累加器。 相反，它只是存储在搜索堆栈中，因此错误受限于 `O(MAX_DEPTH)` 并且大部分可以忽略。

使用量化时这不再是问题，增量实现是一致的，但现在有可能溢出累加器（无论是否使用增量更新）。 必须选择量化方案，使得任何可能的活动特征的组合都不能超过最大值。

### 半KP

HalfKP 是最常见的功能集，其他成功的功能集都建立在它之上。 它适合大小合适的最佳位置，并且平均每次移动需要很少的更新。 每个特征都是一个元组“(our_king_square, piece_square, piece_type, piece_color)”，其中“piece_type”不是国王（在 HalfKA 特征集中包括国王）。 这意味着对于每个国王位置都有一组特征“P”，它们是“(piece_square, piece_type, piece_color)”。 这使网络能够更好地了解与国王相关的棋子。 特征总数为`64*64*5*2=40960`。 （请注意，在当前的 Stockfish 实现中有 Shogi 的遗留问题，还有 64 个未使用的附加功能，但我们将在本文档中忽略它们）。 特征指数可以计算为
```cpp
p_idx = piece_type * 2 + piece_color
halfkp_idx = piece_square + (p_idx + king_square * 10) * 64
```
需要处理的一种特殊情况是国王移动时，因为它与所有特征相关联。 所有功能都已更改，因此执行累加器刷新。 这使得国王移动的成本更高，但平均而言，它仍然保持每次评估的更新数量较低。

现在，您可能会问，“但是哪个国王？！”。 答案是两者...

#### 多视角，多累加器

这是我们需要开始分别考虑双方特征的地方。 白方将保留自己的累加器，黑方也将保留自己的累加器。 实际上，这意味着特征的最大活动数量是只有一个视角的简单特征集的两倍。 更新次数将增加一倍，累加器总数将增加一倍，但总的来说，速度和准确性之间的这种权衡是值得的。 这种方法不可避免地会产生一些关于精确模型拓扑的问题、选项和选择。 让我们一一过一遍。

##### 如何组合多个累加器视角？

由于我们现在有两个累加器，我们需要以某种方式将它们组合成一个向量，进一步传递到网络中。 这可以通过两（三）种方式解决。 让我们将白色的累加器表示为“A_w”，将黑色的累加器表示为“A_b”。

1. 连接 `A_w` 和 `A_b`，首先放置 `A_w`，然后放置 `A_b`。 这是最简单的选择。 这种情况下的输出总是相对于白人的视角。
2. 连接 `A_w` 和 `A_b`，如果要移动是白色的则先放置 `A_w`，否则先放置 `A_b`，然后放置另一个累加器。 这种方法的优点是网络可以学习速度。 它现在知道轮到谁了，这是国际象棋中的一个重要因素，可以对某些位置的评估产生巨大影响。 这种情况下的输出总是相对于要移动透视图的一侧。
3. 1 或 2，但不是连接交错。 所以`A_w[0], A_b[0], A_w[1], A_b[1], ...`。 这在某些并不总是使用整个组合累加器的奇特体系结构中可能是有利的，在这种情况下，交错意味着所使用的切片始终包含相同数量的来自白方和黑方的输出。 这可能会变得有用，例如，当对第一个隐藏层使用结构化稀疏性时，它最终作用于累加器的子集。

##### 每个视角使用哪一组权重？

所以我们计算白色和黑色的特征相同，它们的权重是否相关？ 他们可以，但这不是必需的。 引擎对此的处理有所不同。

1. 两种观点的权重相同。 这意味着棋盘状态需要以某种方式进行定向。 否则，E1 上的白王将产生与 E8 上的黑王不同的特征子集，而 G4 上的白王将产生与 G4 上的黑王相同的特征子集。 那很糟。 解决方案是镜像位置并将棋子的颜色换成黑色的透视图； 那么特征映射的块放置对两者都是合乎逻辑的。 从白方的角度来看，E1 上的白王应该与从黑方的角度来看，E8 上的黑王是一样的。 现在您可能认为翻转是正确的选择，但国际象棋具有垂直对称性，而将棋具有旋转对称性。 Stockfish 中 HalfKP 的初始实现使用旋转来改变视角，这对于国际象棋来说可能是不正确的，但效果出奇地好。
2. 不同视角的权重不同。 E1 上的白王实际上等于 E8 上的黑王吗？ 其他作品呢？ 可以说，黑人和白人玩游戏的方式不同，似乎对这些观点使用不同的特征是有意义的。 一些引擎就是这样做的，这并没有错。 唯一的缺点是尺寸较大，训练时间稍长，但除此之外可能会更好！ 它还完全删除了关于翻转或旋转的讨论，这使得实现更简单。

#### HalfKP示例和网络图

与上图“A”功能集类似，下图是同一网络的图表，但具有 HalfKP 功能集和组合权重。 由于两个累加器的大小都为 4，所以网络最终是`HalfKP[40960]->4x2->8->1`

让我们看一下与之前相同的示例位置：`1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1`。

![](img/board_0.png)

现在我们有两个视角，将分别列出它们的特征。 请记住，特征是`(our_king_square, piece_square, piece_type, piece_color)`，我们使用翻转将方块定向为黑色，颜色反转！ （可以将“颜色”视为“我们”或“他们”）

白方视角：`(A1, C3, pawn, white)`, `(A1, D4, rook, black)`

黑方视角：`(B1, C6, pawn, black)`, `(B1, D5, rook, white)`

网络图现在看起来更有趣了。

![HalfKP[40960]->4x2->8->1](img/HalfKP-40960-4x2-8-1.svg)

## 前向传递实现

在这一部分中，我们将研究模型推理，因为它可以在一个简单的国际象棋引擎中实现。 为了简单起见，我们将在这里使用浮点值。 输入生成超出了此实现的范围。

### 示例网络

我们将采用一个更普遍定义的网络，其架构为“FeatureSet[N]->M*2->K->1”。 因此，这些层将是：

1. `L_0`：线性`N->M`
2. `C_0`: 大小为 `M*2` 的裁剪 ReLU
3. `L_1`：线性`M*2->K`
4. `C_1`：大小为 `K` 的裁剪 ReLU
5. `L_2`：线性`K->1`

###图层参数

线性层有 2 个参数——权重和偏差。 我们将它们分别称为“L_0.weight”和“L_0.bias”。 这些层还包含输入和输出的数量，分别在“L_0.num_inputs”和“L_0.num_outputs”中。

关于权重矩阵的布局，这里必须说一些重要的事情。 对于稀疏乘法，列优先（列在内存中是连续的）布局是有利的，因为我们要添加列，但对于密集乘法，这不是那么清楚，行优先布局可能更可取。 现在我们将坚持列优先布局，但在量化和优化方面，我们可能会重新考虑行优先布局。 现在我们假设 `L_0.weight` 允许访问以下形式的单个元素：`L_0.weight[column_index][row_index]`。

提供的代码非常接近 C++，但可能会省略技术细节。

### 累加器

累加器可以由一个数组表示，该数组与搜索堆栈上的其他位置状态信息一起存储。

```cpp
结构 NnueAccumulator {
     // 两个大小为 M 的向量。v[0] 表示白色的视角，v[1] 表示黑色的视角。
     浮动 v[2][M]；

     // 这将在后面的代码片段中使用，以减少访问的冗长
     float* 运算符[]（颜色透视）{
         返回 v[透视]；
     }
};
```

累加器可以在评估时延迟更新，也可以在每次移动时更新。 这在这里无关紧要，但必须*以某种方式*对其进行更新。 延迟更新还是急切更新更好取决于搜索期间完成的评估次数。 对于更新，有两种情况，如前所述：

1.累加器必须从头开始重新计算。
2. 以前的累加器被重用，只是更新了变化的特性

#### 刷新累加器

```cpp
void refresh_accumulator(
     const LinearLayer& layer, // 这总是 L_0
     NnueAccumulator& new_acc, // 结果存储
     const std::vector<int>& active_features, // 该位置的活动特征索引
     Color perspective // 要刷新的透视图
) {
     // 首先我们复制图层偏差，这是我们的起点
     对于 (int i = 0; i < M; ++i) {
         new_acc[perspective][i] = layer.bias[i];
     }

     // 然后我们只是累加活动特征的所有列。 这就是蓄能器的作用！
     对于（int a：active_features）{
         对于 (int i = 0; i < M; ++i) {
             new_acc[perspective][i] += layer.weight[a][i];
         }
     }
}
```

####更新累加器

```cpp
无效 update_accumulator(
     const LinearLayer& layer, // 这总是 L_0
     NnueAccumulator& new_acc, // 很高兴已经为
                                               // 新的累加器。 相关部分将被覆盖
     const NNueAccumulator& prev_acc, // 前一个累加器，我们正在重用的累加器
     const std::vector<int>& removed_features, // 被移除特征的索引
     const std::vector<int>& added_features, // 添加的特征索引
     颜色透视 // 要更新的透视，记住我们有两个，
                                               // 他们有单独的功能列表，甚至可能会发生
                                               // 一个被更新而另一个需要完全刷新
) {
     // 首先我们复制之前的值，这是我们的起点
     对于 (int i = 0; i < M; ++i) {
         new_acc[透视][i] = prev_acc[透视][i];
     }

     // 然后我们减去移除特征的权重
     对于（int r：removed_features）{
         对于 (int i = 0; i < M; ++i) {
             // 只减去第 r 列
             new_acc[perspective][i] -= layer.weight[r][i];
         }
     }

     // 与添加的功能类似，但添加而不是减去
     对于（int a：added_features）{
         对于 (int i = 0; i < M; ++i) {
             new_acc[perspective][i] += layer.weight[a][i];
         }
     }
}
```

就是这样！ 很简单，不是吗？

### 线性层

这是简单的向量矩阵乘法，你问它有什么复杂的？ 现在什么都没有，但是一旦优化开始，它就会变得复杂。 现在我们不会优化，但我们至少会编写一个使用权重矩阵具有列优先布局这一事实的版本。

```cpp
浮动*线性（
     const LinearLayer& layer, // 要使用的图层。 我们有两个：L_1，L_2
     float* output, // 已经为结果分配的存储空间
     const float* input // 输入，即前一个 ClippedReLU 层的输出
) {
     // 首先将偏差复制到输出。 我们将在其上添加列。
     对于 (int i = 0; i < layer.num_outputs; ++i) {
         output[i] = layer.bias[i];
     }

     // 还记得很久以前的彩虹图吗？ 就是这个。
     // 我们正在逐一添加列，按输入值缩放。
     对于 (int i = 0; i < layer.num_inputs; ++i) {
         对于 (int j = 0; j < layer.num_outputs; ++j) {
             输出[j] += 输入[i] * layer.weight[i][j];
         }
     }

     // 让调用者知道使用的缓冲区在哪里结束。
     返回输出+layer.num_outputs；
}
```

### ClippedReLU

```cpp
浮动*克雷卢（，
     int size, // 不需要任何层结构，我们只需要元素的数量
     float* output, // 已经为结果分配的存储空间
     const float* input // 输入，即前一个线性层的输出
) {
     for (int i = 0; i < size; ++i) {
         输出[i] = min(max(输入[i], 0), 1);
     }

     返回输出+大小；
}
```

＃＃＃ 把它放在一起

在粗略的伪代码中。 特征索引生成留给读者作为练习。

```cpp
void Position::do_move(...) {
     ... // 做移动的东西

     对于（颜色透视：{白色，黑色}）{
         如果（需要刷新[透视]）{
             刷新累加器（
                 L_0,
                 这个->累加器，
                 这个->get_active_features（透视），
                 看法
             );
         } 别的 {
             更新累加器（
                 L_0,
                 这个->累加器，
                 this->get_previous_position()->累加器，
                 这个->get_removed_features（透视），
                 这个->get_added_features（透视），
                 看法
             );
         }
     }
}

float nnue_evaluate(const Position& pos) {
     浮动缓冲区[...]； // 为结果分配足够的空间

     // 我们需要先准备输入！ 我们将把蓄能器用于
     // 一侧先移动，另一侧移动。
     浮动输入[2*M]；
     颜色 stm = pos.side_to_move;
     对于 (int i = 0; i < M; ++i) {
         输入[i] = pos.accumulator[stm][i];
         输入[M+i] = pos.accumulator[!stm][i];
     }

     float* curr_output = 缓冲区；
     浮动* curr_input = 输入；
     浮动*下一个输出；

     // 评估一层并向前移动输入和输出。
     // 最后一个输出成为下一个输入。
     next_output = crelu(2 * L_0.num_outputs, curr_output, curr_input);
     当前输入 = 当前输出；
     当前输出 = 下一个输出；

     next_output = linear(L_1, curr_output, curr_input);
     当前输入 = 当前输出；
     当前输出 = 下一个输出；

     next_output = crelu(L_1.num_outputs, curr_output, curr_input);
     当前输入 = 当前输出；
     当前输出 = 下一个输出；

     next_output = linear(L_2, curr_output, curr_input);

     // 我们完成了。 最后一层应该在 *curr_output 下放 1 个值。
     返回 *curr_output；
}
```

就是这样！ 这就是整个网络。 你说你不能用它是什么意思？！ 哦对了，你没有训练过的网，真可惜。

## 用 pytorch 训练网络

这将非常简短，因为毕竟它在 nnue-pytorch 存储库中，所以您可以直接查找代码！ 我们不会解释 pytorch 是如何工作的，但是我们会解释一些基础知识，以及适应这种奇异用例所需的怪癖。

让我们继续使用前向传递实现中的架构。

### 型号说明

Pytorch 内置了线性层的类型，因此定义模型非常简单。

```python
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.ft = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, K)

    # 输入的是整批！
    # `stm` 表示白棋是否是要走的一方。 1 = 真，0 = 假。
    def forward(self, white_features, black_features, stm):
        w = self.ft(white_features) # white's perspective
        b = self.ft(black_features) # black's perspective

        # 请记住，我们根据谁要移动来对 2 个视角的累加器进行排序。
        # 因此，我们通过在“stm”和“1-stm”张量之间插值来混合两种可能的顺序。
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # 运行线性层并使用 clamp_ 作为 ClippedReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        return self.l2(l2_x)
```

值得庆幸的是，Pytorch 通过自动微分自动处理反向传播。 整洁的！ 现在的难点在于，也许令人惊讶的是，提供数据。

###准备输入

这部分主要有两个瓶颈。

1.解析训练数据集
2.准备张量输入

#### 解析训练数据集并移至python端

您可能想在 python 中实现它。 它会起作用，但遗憾的是，它会慢几个数量级。 我们在 nnue-pytorch 中所做的是我们用 C++ 创建了一个共享库，它实现了一个非常快速的训练数据解析器，并以可以快速转换为输入张量的形式提供数据。

我们将使用 [Ctypes](https://docs.python.org/3/library/ctypes.html) 来实现 C 和 Python 之间的互操作。 [Seer 的培训师](https://github.com/connormcmonigle/seer-training/tree/6077a044c596963a34c504df8450aceaaa2b3fb1) 如果您需要更多示例，请使用 pybind11。 实际上，任何提供从 Python 传递指针和调用 C 函数的方法都可以。 也可以使用其他语言，但请记住，只有 C 具有稳定的 ABI，这使事情变得更容易和更便携。 因此，例如，如果您想使用 C++（就像我们将在此处使用的那样），将导出的函数标记为 extern "C" 很重要。

数据读取器在创建时传递一个文件，然后它生成请求数量的工作线程，这些线程咀嚼数据并异步准备**整批**。 然后将批次传递到 python 端并转换为 PyTorch 张量。 一个样品一个样品地生产不是一个可行的选择，需要通过生产整批来削减角落。 你可能会问为什么？ PyTorch 可以将多个张量变成一个批处理，那有什么问题呢？ 让我们来看看...

还记得输入是如何稀疏的吗？ 现在假设我们的批量大小是 8192。如果我们发送 8192 个稀疏张量并尝试从中形成一个批量会发生什么？ 好吧，pytorch 不喜欢自己做那件事，我们需要帮助它。 最好的方法是形成一个包含整个批次的大二维稀疏输入张量。 它有 2 个稀疏维度，索引是“(position_index, feature_index)”，非常简单，性能很好，而且不需要创建临时张量！ 我们从一开始就形成整个批次的事实也意味着我们可以减少分配量并为批次部分使用更好的内存布局。

因此，我们也不能简单地使用 PyTorch 的“DataLoader”，而是需要将其用作单纯的包装器。 但这种努力是值得的。 一个工作线程通常可以毫无问题地使高端 GPU 饱和。

#### 训练批处理结构和通信

至少需要的是特征（从两个角度）、要移动的边（用于累加器切片排序）和位置评估（分数）。 让我们看看如何表示这样的一批。

```cpp
struct SparseBatch {
    SparseBatch(const std::vector<TrainingDataEntry>& entries) {

        // The number of positions in the batch
        size = entries.size();

        // The total number of white/black active features in the whole batch.
        num_active_white_features = 0;
        num_active_black_features = 0;

        // The side to move for each position. 1 for white, 0 for black.
        // Required for ordering of the accumulator slices in the forward pass.
        stm = new float[size];

        // The score for each position. This is value that we will be teaching the network.
        score = new float[size];

        // 活动特征的索引。
        // 为什么大小要乘以 2？答案是这些索引是二维的
        //（位置索引，特征索引）。它实际上是一个矩阵，大小为
        //（num_active_*_features, 2）。
        // 重要：我们必须确保索引是按升序排列的。
        // 也就是说，首先是第一个位置，然后是第二个，然后是第三个，
        // 以此类推。而在一个位置的特征中，特征索引
        // 也必须按照升序排列。为什么需要这样做稍后将会变得明显。
        white_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];
        black_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];

        fill(entries);
    }

    void fill(const std::vector<TrainingDataEntry>& entries) {
        ...
    }

    int size;
    int num_active_white_features;
    int num_active_black_features;

    float* stm;
    float* score;
    int* white_features_indices;
    int* black_features_indices;

    ~SparseBatch()
    {
        // RAII! Or use std::unique_ptr<T[]>, but remember that only raw pointers should
        // be passed through language boundaries as std::unique_ptr doesn't have stable ABI
        delete[] stm;
        delete[] score;
        delete[] white_features_indices;
        delete[] black_features_indices;
    }
};
```

and in python

```python
class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('stm', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('white_features_indices', ctypes.POINTER(ctypes.c_int)),
        ('black_features_indices', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self, device):
		# 这是说明性的。 实际上你可能需要转移这些
        # 到 GPU。 您也可以异步执行，但请记住
        # 确保源存在足够长的时间以使副本完成。
        # 请参阅 torch.tensor.to(...) 了解更多信息。

        # 这是将指针转换为 pytorch 张量的好方法。
        # 需要传递形状，记住我们正在形成整批，第一个
        # 维度始终是批量大小。
        stm_t = torch.from_numpy(
            np.ctypeslib.as_array(self.stm, shape=(self.size, 1)))
        score_t = torch.from_numpy(
            np.ctypeslib.as_array(self.score, shape=(self.size, 1)))

		# 正如我们所说，索引张量需要转置（不是整个稀疏张量！）。
        # 这就是 pytorch 在稀疏张量中存储索引的方式。
        # 它还要求索引为 64 位整数。
        white_features_indices_t = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.white_features_indices, shape=(self.num_active_white_features, 2))
            ), 0, 1).long()
        black_features_indices_t = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.black_features_indices, shape=(self.num_active_white_features, 2))
            ), 0, 1).long()

        # 这些值都是 1，因此我们可以轻松地创建这些张量。
        # 无需通过副本。
        white_features_values_t = torch.ones(self.num_active_white_features)
        black_features_values_t = torch.ones(self.num_active_black_features)

		# 现在是魔法。 我们通过给出索引来构造稀疏张量
        # 非零值（活动特征索引）和值本身（全是！）。
        # 张量的大小为batch_size*NUM_FEATURES，这将
        # 通常会非常大，但由于密度约为 0.1%，因此需要
        # 空间很小，可以更快地向前传球。
        # 为了获得最佳性能，我们确实做了一些作弊。 通常是火炬
        # 检查正确性，这是一个昂贵的 O(n) 操作。
        # 通过使用 _sparse_coo_tensor_unsafe 我们可以避免这种情况。
        white_features_t = torch._sparse_coo_tensor_unsafe(
            white_features_indices_t, white_features_values_t, (self.size, NUM_FEATURES))
        black_features_t = torch._sparse_coo_tensor_unsafe(
            black_features_indices_t, black_features_values_t, (self.size, NUM_FEATURES))

		# 什么是合并？！ 它确保索引是唯一且有序的。
        # 现在你可能明白为什么我们说输入必须从一开始就排序。
        # 这通常是一个 O(n log n) 操作并且需要大量的时间
        ＃ 时间。 但在这里我们**知道**张量已经处于合并形式，
         # 因此我们可以告诉 pytorch 它可以使用这个假设。
        white_features_t._coalesced_(True)
        black_features_t._coalesced_(True)

        # Now this is what the forward() required!
        return white_features_t, black_features_t, stm_t, score_t

# 我们还告诉 ctypes 如何理解这种类型。
SparseBatchPtr = ctypes.POINTER(SparseBatch)
```

### 特征分解

让我们再次关注这些功能。 我们将仔细研究“HalfKP”功能集。 嗯...我们取了`P`，做了 64 次，每个方格一次...这 64 个桶肯定有某种关联...我们如何告诉网络它们是相关的？ 通过引入虚拟功能！

我们有 40960 个“HalfKP”特征和 640 个“P”特征。 他们如何相互映射？ 确切的计算将取决于您的索引方案，但我们可以用简单的术语进行布局。

`HalfKP` 特征是 `(king_square, piece_square, piece_type, piece_color)`

`P` 特征是 `(piece_square, piece_type, piece_color)`。

两者共有 3 个部分。 因此，对于每个“P”特征，都有 64 个对应的“HalfKP”特征。 我们可以将 40960 输入扩展到 40960+640，包括“HalfKP”和“P”功能。 现在每个位置每个视角最多有 64 个特征（32 个“HalfKP”和 32 个“P”）。 数据加载器和前向传递没有其他变化，我们只是添加了更多功能！ 但我们不想在实际比赛中使用它们，那样太贵了，而且有点毫无意义。 我们知道哪些特征是相互关联的，所以让我们在使用网络进行游戏之前以某种方式合并它们。

#### 虚拟特征合并

那么我们如何合并它们呢？ 让我们再看看矩阵和向量乘法是如何完成的。 考虑之前的示例位置 (`1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1`)。

![](img/board_0.png):

让我们关注特征`(A1, C3, pawn, white)`。 现在，我们还要添加一个 `P` 特征 `(C3, pawn, white)`。 当输入通过第一层时会发生什么？

```cpp
累加器 += 权重 [(A1, C3, pawn, white)];
累加器 += 权重 [(C3, pawn, white)];
```

这相当于

```cpp
累加器 += 权重[(A1, C3, pawn, white)] + weights[(C3, pawn, white)];
```

所以关系很简单。 我们只需要将每个 `P` 特征的权重添加到所有相关的 `HalfKP` 特征权重中！

#### 其他因素

有时可以添加更多因素。 但需要注意的是，仅仅增加更多因素并不一定会改善训练，甚至可能导致其倒退。 一般来说，使用某些因素是否有帮助取决于训练设置和被训练的网络。 试验这些东西总是好的。 尽管如此，我们可以考虑例如“HalfKP”的以下因素。

#####“K”因素

王位，64个特征。 这需要小心处理，因为一个位置多次具有此功能 - 棋盘上的棋子数量。 这意味着此功能的输入不再是 1，而是棋盘上的位置数。 这纯粹是需要的，因为使用 HalfKP，国王特征不会在任何地方编码。 例如，HalfKA 不需要它，因为它专门具有国王位置的功能。 一般来说，处理这个很棘手，它甚至可能需要降低这些特征的梯度（否则梯度是“input*weight”，但与其他输入相比输入较大）。

#####“HalfRelativeKP”因素

在“HalfKP”中，我们使用绝对棋子位置，但如果我们将位置编码为相对于国王的位置呢？ 有 15x15 种可能的相对位置，其中大部分对应于 1:many 某些“HalfKP”特征。 HalfRelativeKP 特征索引可以这样计算，例如：
```cpp
int get_half_relative_kp_index(颜色透视图, Square king_sq, Square piece_sq, Piece piece)
{
     const int p_idx = static_cast<int>(piece.type()) * 2 + (piece.color() != perspective);
     const Square oriented_king_sq = orient_flip(perspective, king_sq);
     const Square oriented_piece_sq = orient_flip(perspective, piece_sq);
     // 文件/排名差异始终在 -7..7 范围内，我们需要将其映射到 0..15
     const int relative_file = oriented_piece_sq.file() - oriented_king_sq.file() + 7;
     const int relative_rank = oriented_piece_sq.rank() - oriented_king_sq.rank() + 7;
     返回 (p_idx * 15 * 15) + (relative_file * 15) + relative_rank;
}
```

#### 分解器的真实效果

虽然因子分解器帮助网络进行泛化，但它似乎只在早期阶段相关，即当网络还什么都不知道的时候。 它加速了训练的早期阶段并减少了输入的稀疏性（否则一些输入非常罕见）。 但它很快就变得不重要了，在训练的后期可以删除它以获得一些训练速度（毕竟它可以添加很多活跃的特征）。

### 损失函数以及如何应用它们

＃＃＃＃ 目标

训练网络实际上只是最小化损失函数，它需要平滑并且在“最佳”评估（训练目标）处具有最小值。 就NNUE而言，这是通过通常的机器学习方法（也有非梯度方法，这里不做描述）通过梯度下降来完成的。

#### 将评估从 CP 空间转换为 WDL 空间

我们所说的 CP 空间是指厘泊比例（或某种成比例的东西，例如引擎的内部单位）。 对于 WDL 空间，我们的意思是 0 = 输，0.5 = 平局，1 = 赢。

当然可以将损失函数直接应用于评估值（在 CP 空间中），但这会导致大梯度（或大量超参数调整），限制可用的损失函数集，并且不允许 使用结果作为损失。 我们将专注于 WDL 空间中的评估。 但是如何在这些空间之间进行转换呢？ 通常对性能对应的评估可以很好地用 sigmoid 拟合。 例如，在 Stockfish 生成的一些数据中，我们有：

![](img/sigmoid_wdl_fit.png)


所以在代码中我们可能会做以下事情：
```蟒蛇
scaling_factor = 410 # 这取决于引擎，甚至可能取决于数据
wdl_space_eval = torch.sigmoid(cp_space_eval / scaling_factor)
```

这种转变还有一个很好的效果，即大型评估变得“更紧密”在一起，这与实际游戏非常吻合，大型评估不需要那么精确。

#### 在评估中使用结果

由于我们将计算损失的值位于 WDL 空间中，我们现在可以将它们与游戏结果进行插值。 我们将引入一个控制插值的“lambda_”参数。
```蟒蛇
# game_result 在 WDL 空间中
wdl_value = lambda_ * wdl_space_eval + (1 - lambda_) * game_result
```

插值也可以应用于损失。
```蟒蛇
loss_eval = ... # 模型评估和位置评估之间的损失
loss_result = ... # 模型评估和游戏结果之间的损失
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
```

哪种方式效果更好取决于你的情况:)

#### 均方误差 (MSE)

现在我们知道我们要适应什么了； 让我们看看我们将如何适应他们。

这是一个非常简单的损失函数，只取预测值和目标之间差异的平方。 这导致了一个很好的线性渐变。

之前应用插值：
```python
scaling = ... # 取决于引擎和数据。 确定形状
               # 将评估转换为 WDL 空间的 sigmoid
               # Stockfish 使用大约 400 的值
wdl_eval_model = sigmoid（模型（...）/缩放）
wdl_eval_target = sigmoid（目标/缩放）
wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result
损失 = (wdl_eval_model - wdl_value_target)**2
```

在之后应用插值：
```python
缩放 = ...
wdl_eval_model = sigmoid（模型（...）/缩放）
wdl_eval_target = sigmoid（目标/缩放）
loss_eval = (wdl_eval_model - wdl_eval_target)**2
loss_result = (wdl_eval_model - game_result)**2
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
```

注意：实际上指数可以 >2。 更高的指数以准确性为代价给予精度更多的权重。 Stockfish 网络具有良好的训练结果，例如指数为 2.6。

＃＃＃＃＃ 损失

![](img/mse_loss.png)
![](img/mse_loss_contour.png)

#####毕业

![](img/mse_loss_grad.png)
![](img/mse_loss_grad_contour.png)

#### 交叉熵

这个损失函数通常用于连续分类问题，我们的用例可以被认为是一个。

必须注意域边界。 通常会添加一个非常小的值 (epsilon)，这样这些值在对数下永远不会达到 0。

之前应用插值：

```python
epsilon = 1e-12 # to prevent log(0)
scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)
wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result

# 损失中的第一项的梯度为 0，因为我们总是
# 相对于“wdl_eval_model”进行区分，但它使损失变得更好
# 0 是最小值。
loss = (wdl_value_target * log(wdl_value_target + epsilon) + (1 - wdl_value_target) * log(1 - wdl_value_target + epsilon))
      -(wdl_value_target * log(wdl_eval_model   + epsilon) + (1 - wdl_value_target) * log(1 - wdl_eval_model   + epsilon))
```

在之后应用插值：

```python
epsilon = 1e-12 # to prevent log(0)
scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)

# 损失中的第一项的梯度为 0，因为我们总是
# 相对于“wdl_eval_model”进行区分，但它使损失变得更好
# 0 是最小值。
loss_eval   = (wdl_eval_target * log(wdl_eval_target + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_target + epsilon))
             -(wdl_eval_target * log(wdl_eval_model  + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_model  + epsilon))
loss_result = (game_result     * log(wdl_eval_target + epsilon) + (1 - game_result)     * log(1 - wdl_eval_target + epsilon))
             -(game_result     * log(wdl_eval_model  + epsilon) + (1 - game_result)     * log(1 - wdl_eval_model  + epsilon))
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
```

＃＃＃＃＃ 损失

![](img/cross_entropy_loss.png)
![](img/cross_entropy_loss_contour.png)

#####毕业

![](img/cross_entropy_loss_grad.png)
![](img/cross_entropy_loss_grad_contour.png)

## 量化

在本文档的开头，简要提到了量化是什么以及它的重要性。 现在是正确理解它的时候了。 目标是我们希望在任何地方都使用尽可能小的整数。 大多数 CPU 架构提供的指令可以同时处理 8、16、32 甚至 64 个 int8 值，我们应该利用这一点。 这意味着我们需要使用范围为 -128..127 的 int8 值作为权重和输入； 或 int16，范围为 -32768..32767，其中 int8 是不可能的。

想出正确的量化方案并不容易，所以首先我们将介绍 Stockfish 当前使用的方案，然后我们将解释如何实现、如何编码，最后如何优化它。

### Stockfish量化方案

#### 特征转换器

让我们从特征转换器开始。 回想一下，它的目的是累积 0 到 30（对于 HalfKP）行的权重。 我们希望将 int8 值作为后面层的输入，激活范围 (ClippedReLU) 为 0..127，但这意味着使用 int8 整数作为累加器不能提供足够的空间，因为值会超出范围 在应用 ClippedReLU 之前使用 int8...所以我们使用 int16 作为累加器，然后在执行 ClippedReLU 时转换为 int8。

#### 线性层

我们想要 int8 输入并且我们可以在不损失太多精度的情况下获得它们。 用于矩阵的 SIMD 指令的本质是，幸运的是，累加发生在 int32 中。 因此，我们不会遇到与手动添加行的特征转换器相同的问题，我们可以最大程度地利用 int8 乘法和 int32 累加，然后才返回到 ClippedReLU 层中的 int8。 我们将在累积发生后添加偏差，因此它们应该存储在 int32 中。

#### ClippedReLU

这里没什么特别的。 由于输入没有被缩放，这只是相同的操作，但在不同的域中。 我们没有固定到 0..1，而是固定到 0..127。 输入类型通常不同于输出类型，因为输入是 int32 或 int16，而我们想要的输出是 int8。 值不会改变，但需要应用转换。

### 量化的数学以及如何使其适合

为了量化网络，我们需要将权重和偏差乘以某个常数，以将它们转换为不同的值范围。 这在网络推理过程中遇到乘法运算时会带来问题 - `(a*x) * (a*w) = a*a*x*w`，我们有时也必须缩减输出。 但是每一层仍然是独立的，所以让我们一层一层地看一遍。

#### 特征转换器

请记住，我们希望激活范围从 0..1 变为 0..127。 由于特征变换器是一个纯加法过程，我们将权重和偏差乘以 127 就足够了。权重和偏差都存储为 int16。 我们可以将输出除以某个因子“a”以获得更高的精度，在这种情况下，权重和偏差必须乘以“a*127”，但实际上它只会提高一点点精度。

#### 线性层

为了达到 int8 权重，我们必须应用一些比例因子。 这个比例因子最终取决于需要保留多少精度，但不能太大，因为这样权重的大小就会受到限制。 例如，如果我们将比例因子设为 64（在 Stockfish 中使用），则浮点空间中的最大权重为 127/64=1.984375。 这足以拥有良好的网，但在训练期间需要注意夹紧重量，以免它们超出该范围。 比例因子 64 也可以理解为可以表示为“1/64=0.015625”的最小权重步长。

线性层只是矩阵乘法，所以我们将输入和权重相乘，但现在两者都相对于浮点版本进行了缩放。 让我们将输入缩放因子（激活范围缩放）表示为“s_A”，将权重缩放因子表示为“s_W”。 x 是未量化的输入，w 是未量化的权重，b 是未量化的偏差，y 是未量化的输出。
所以我们有：

```
x * w + b = y
((s_A * x) * (s_W * w)) + (b * s_A * s_W) = (y * s_A) * s_W
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) / s_W = (y * s_A)
```
从中我们了解到，我们需要通过 `(s_A * s_W)` 缩放偏差，通过 `s_W` 缩放权重，并将输出除以 `s_W` 以获得所需的 `(y * s_A)`，它被正确缩放到 激活范围。

现在，这仅适用于下一层是 ClippedReLU 层的情况。 对于最后一层，输出范围非常不同，量化也会不同。 在 Stockfish 中，我们希望最后一层输出 -10000..10000 范围内的值，同时仍保持 int8 权重。 这可以在没有任何额外比例因子的情况下实现，但使用额外的比例因子最容易做到和理解。

我们将引入一个新的比例因子 `s_O`。 与其他比例因子不同，这个比例因子需要在训练（针对实际评估进行损失计算）和推理期间应用于输出。 它的目的是缩放网络的浮点输出以匹配 Stockfish 使用的整数评估范围。 基本上这意味着浮动空间中的“1”等于“s_O”内部评估单元。 它还有一个额外的优势，那就是它允许我们让层的权重在大小上与之前的层相似。

所以现在的数学是：

```
x * w + b = y
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O = ((y * s_A) * s_W) * s_O
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O / s_A / s_W = (y * s_O)
(((s_A * x) * (s_W / s_A * w)) + (b * s_A * s_W / s_A)) * s_O / s_W = (y * s_O)
(((s_A * x) * (s_W * s_O / s_A * w)) + (b * s_W * s_O)) / s_W = (y * s_O)
```
从中我们了解到，我们需要通过 `s_W * s_O` 缩放偏差，通过 `s_W * s_O / s_A` 缩放权重，并将输出除以 `s_W` 以获得所需的 `(y * s_O)`。

＃＃＃ 执行

对于未优化的实现，没有太多变化。 只需要记住将数据类型更改为具有所需大小的整数、缩放输入的权重，并将线性层的输出除以“s_W”。 `s_W` 通常选择为 2 的幂，因此这个操作是一个简单的按位右移，因为没有用于整数的 SIMD 除法指令，即使有它也会很慢。

### 优化实现

为简单起见，我们将只关注 x86-64 指令集的 AVX2 扩展的优化。

#### 特征转换器

SIMD 对特征转换器的好处有两个：

1. 一条指令可以执行多次加法
2. 较大的总寄存器大小意味着我们不需要经常写入内存

我们的累加结构没有太大变化，我们只是将float改为int16：
```cpp
// 我们现在还要确保累加器结构与缓存行对齐。
// 这不是 AVX2 指令的严格要求，但可能会提高性能。
结构 alignas(64) NnueAccumulator {
     // 两个大小为 N 的向量。v[0] 表示白色的视角，v[1] 表示黑色的视角。
     int16_t v[2][N]；

     // 这将在后面的代码片段中使用，以减少访问的冗长
     int16_t* 运算符[]（颜色透视）{
         返回 v[透视]；
     }
};
```

现在让我们看一下刷新函数。 为简单起见，我们假设有足够的寄存器以便不会发生溢出，但实际上（`M > 256`）需要对活动特征进行多次传递，每次只考虑累加器的一部分。 单个 AVX2 寄存器可以容纳 16 个 int16 值，并且有 16 个 AVX2 寄存器（自 AVX-512 以来为 32 个）。

```cpp
void refresh_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // storage for the result
    const std::vector<int>& active_features,  // the indices of features that are active for this position
    Color                   perspective       // the perspective to refresh
) {
    // The compiler should use one register per value, and hopefully
    // won't spill anything. Always check the assembly generated to be sure!
    constexpr int register_width = 256 / 16;
    static_assert(M % register_width == 0, "We're processing 16 elements at a time");
    constexpr int num_chunks = M / register_width;
    __m256i regs[num_chunks];

    // Load bias to registers and operate on registers only.
    for (int i = 0; i < num_chunks; ++i) {
        regs[i] = _mm256_load_si256(&layer.bias[i * register_width]);
    }

    for (int a : active_features) {
        for (int i = 0; i < num_chunks; ++i) {
            // Now we do 1 memory operation instead of 2 per loop iteration.
            regs[i] = _mm256_add_epi16(regs[i], _mm256_load_si256(&layer.weight[a][i * register_width]));
        }
    }

    // Only after all the accumulation is done do the write.
    for (int i = 0; i < num_chunks; ++i) {
        _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]);
    }
}
```

同样的更新：

```cpp
void update_accumulator(
    const LinearLayer&      layer,            // 这将始终是 L_0
    NnueAccumulator&        new_acc,          // 最好已为新的累加器提供存储空间，相关部分将被覆盖
    const NNueAccumulator&  prev_acc,         // 我们要重用的前一个累加器
    const std::vector<int>& removed_features, // 被移除的特征的索引
    const std::vector<int>& added_features,   // 被添加的特征的索引
    Color                   perspective       // 要更新的视角，注意我们有两个视角，它们有各自独立的特征列表，可能一个在更新，另一个需要完全刷新
) {
    // 编译器应为每个值使用一个寄存器，并希望不会溢出任何内容。始终检查生成的汇编以确保准确无误！
    constexpr int register_width = 256 / 16;
    static_assert(M % register_width == 0, "我们一次处理 16 个元素");
    constexpr int num_chunks = M / register_width;
    __m256i regs[num_chunks];

    // 将之前的值加载到寄存器中，仅在寄存器上进行操作
    for (int i = 0; i < num_chunks; ++i) {
        regs[i] = _mm256_load_si256(&prev_acc[perspective][i * register_width]);
    }

    // 然后我们减去被移除特征的权重
    for (int r : removed_features) {
        for (int i = 0; i < num_chunks; ++i) {
            regs[i] = _mm256_sub_epi16(regs[i], _mm256_load_si256(&layer.weight[r][i * register_width]));
        }
    }

    // 对于被添加的特征，我们做类似的操作，但是是添加而不是减去
    for (int a : added_features) {
        for (int i = 0; i < num_chunks; ++i) {
            regs[i] = _mm256_add_epi16(regs[i], _mm256_load_si256(&layer.weight[a][i * register_width]));
        }
    }

    // 只有在所有累积操作完成后才进行写入
    for (int i = 0; i < num_chunks; ++i) {
        _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]);
    }
}
```

#### 线性层

矩阵乘法通常很难优化，根据矩阵的大小有很多方法。 由于我们希望层数很小，因此我们不会深入研究任何奇特的分块算法。 并且仅依靠手动展开并尝试一次处理多个值。 这不是最佳的，但它很简单而且非常接近。 我们只会描述输出数量可以被 4 整除的情况。输出层有 1 个输出，但它也很小，不需要任何聪明的东西。 我们还将要求输入大小是 32 的倍数，否则需要添加 0 填充。

```cpp
int32_t* linear(
    const LinearLayer& layer,  // 要使用的层，我们有两个：L_1，L_2
    int32_t*           output, // 已分配的存储结果的存储空间
    const int8_t*      input   // 输入，这是上一层 ClippedReLU 层的输出
) {
    constexpr int register_width = 256 / 8;
    assert(layer.num_inputs % register_width == 0, "我们一次处理 32 个元素");
    assert(layer.num_outputs % 4 == 0, "我们解卷4个元素");
    const int num_in_chunks = layer.num_inputs / register_width;
    const int num_out_chunks = layer.num_outputs / 4;

    for (int i = 0; i < num_out_chunks; ++i) {
        // 准备权重偏移量。一个偏移量对应一行权重。
        // 这是一个简单的二维数组索引。
        const int offset0 = (i * 4 + 0) * layer.num_inputs;
        const int offset1 = (i * 4 + 1) * layer.num_inputs;
        const int offset2 = (i * 4 + 2) * layer.num_inputs;
        const int offset3 = (i * 4 + 3) * layer.num_inputs;

        // 累加从0开始，我们只在最后添加偏置。
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        // 每个最内层的循环处理一个 32x4 的权重块，所以一次处理 128 个权重！
        for (int j = 0; j < num_in_chunks; ++j) {
            // 我们解卷 4 个元素，这样我们可以重用这个值，从而减少了
            // 所需的内存操作数量。
            const __m256i in = _mm256_load_si256(&input[j * register_width]);

            // 这个函数处理一个 32x1 的 int8 块，并产生一个 8x1 的 int32 块。
            // 对于定义请看下面。
            m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256(&layer.weights[offset0 + j * register_width]));
            m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256(&layer.weights[offset1 + j * register_width]));
            m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256(&layer.weights[offset2 + j * register_width]));
            m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256(&layer.weights[offset3 + j * register_width]));
        }

        const __m128i bias = _mm_load_si128(&layer.bias[i * 4]);
        // 这个函数将每个和中的 8 个值水平相加，生成 4 个 int32 值。
        // 对于定义请看下面。
        __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, bias);
        // 在这里我们考虑了权重缩放。
        outval = _mm_srai_epi32(outval, log2_weight_scale);
        _mm_store_si128(&output[i * 4], outval);
    }

    return output + layer.num_outputs;
}
```

##### m256_add_dpbusd_epi32

![](img/m256_add_dpbusd_epi32.svg)

输出需要进一步水平累加，但稍后用 4 个和（sum0、sum1、sum2、sum3）来做会更快。

此功能可以受益于 VNNI 扩展，此处由“USE_VNNI”控制。

```cpp
void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {
#if defined (USE_VNNI)

    // 这与下面解释的操作完全一样，但只使用了一条指令
    acc = _mm256_dpbusd_epi32(acc, a, b);

#else

    // 将 a * b 的结果进行相邻输出累积成 int16 值
    __m256i product0 = _mm256_maddubs_epi16(a, b);

    // 将 product0 乘以 1（等幂），并将相邻输出累积成 int32 值
    __m256i one = _mm256_set1_epi16(1);
    product0 = _mm256_madd_epi16(product0, one);

    // 添加到主 int32 累加器
    acc = _mm256_add_epi32(acc, product0);

#endif
};
```

##### m256_haddx4

该函数取 4 个 \_\_m256i 寄存器，每个寄存器包含 8 个 int32 值，水平累加它们，并产生一个 \_\_m128i 寄存器，每个寄存器包含 4 个 int32 值，每个寄存器对应一个输入和。 在上面的矩阵乘法中，我们为每个权重行/输入保留一个和，所以最后我们一次填充输出 4 个值。

![](img/m256_haddx4.svg)

```cpp
__m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);

    sum0 = _mm256_hadd_epi32(sum0, sum2);

    __m128i sum128lo = _mm256_castsi256_si128(sum0);
    __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

    return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
};
```

#### 具有稀疏输入的线性层

在前面的部分中，我们描述了通用的密集矩阵乘法，但让我们尝试更深入地研究一下。 我们将在这里考虑的情况类似于我们的特征转换器的操作方式，但在这里我们总是需要执行完整的操作而不是矩阵更小。 但是我们为什么要考虑这个呢？ 好吧，事实证明，在通过 ClippedReLU 之后，特征转换器的输出可以具有相当大的稀疏性。 以下是一些数据，显示了第一个密集全连接层的输入密度，对于具有不同特征变换器大小的网络：

![](img/fc_input_density.png)

（方框对应[25%, 75%]区间，胡须对应[1%, 99%]区间）

对于常见尺寸，这已经 <=15% 密度，并且在不同网络之间是一致的！ 然而，我们无法让它变得更快，因为更改访问模式和需要更多的预处理会产生一些成本，因此这种方法是否适用于您的特定情况需要进行彻底测试。

让我们看看可以利用它的代码。

```cpp
int lsb(std::uint32_t v) {
    // 返回 v 中最低有效位
    // 实现细节
    // 可以使用编译器内部函数来实现
    // https://www.chessprogramming.org/BitScan#Leading_Zero_Count
}

// 这种实现需要改变布局并将权重扩展到 int16。
// 我们现在将转置权重，因为我们现在将沿列而不是行进行处理。
void load_weights(
    const LinearLayer& layer,
    const int8_t* data
) {
    static_assert(is_same_v<LinearLayer::WeightType, int16_t>,
        "这种方法需要权重为 16 位。否则，将乘法输出扩宽到 32 位很困难。");

    for (int i = 0; i < layer.num_outputs; ++i) {
        for (int j = 0; j < layer.num_inputs; ++j) {
            layer.weights[j*layer.num_outputs + i] = data[i*layer.num_inputs + j];
        }
    }

    // 对于 AVX2，我们还必须在权重中交换一些通道。这是
    // 因为 AVX2 寄存器函数作为两个 128 位的寄存器，而且
    // 在推理过程中，一些数据是交错的。
    // 这使得它们最终出现在我们希望的地方。
    // 在可视化中将更加明显。
    // 这实际上是在每个 256 位块中交换出中间的 2 个 64 位块。
    for (int i = 0; i < layer.num_outputs; ++i) {
        for (int j = 0; j < layer.num_inputs; ++j) {
            const int simd_lane = j % 16;
            const int simd_lane_64 = simd_lane / 4;
            if (simd_lane_64 == 1) {
                swap(
                    layer.weights[i*layer.num_outputs + j + 0],
                    layer.weights[i*layer.num_outputs + j + 4]
                );
            }
        }
    }
}

int32_t* linear_sparse_input(
    const LinearLayer& layer,
    int32_t*           output,
    const int8_t*      input
) {
    static_assert(is_same_v<LinearLayer::WeightType, int16_t>,
        "这种方法需要权重为 16 位。否则，将乘法输出扩宽到 32 位很困难。");

    constexpr int register_width = 256 / 8;
    constexpr int input_register_width = register_width; // uint8_t
    constexpr int output_register_width = register_width / 4; // int32_t
    constexpr int output_chunk_size = output_register_width * 2; // 我们一次处理 2 个寄存器
    assert(layer.num_outputs % output_chunk_size == 0, "我们一次处理 16 个输出元素");
    assert(layer.num_inputs % input_register_width == 0);

    // 我们需要找出输入值非零的索引
    uint16_t nnz_input_indices[layer.num_inputs];
    int num_nnz_input_indices = 0;

    for (int i = 0; i < layer.num_inputs; i += input_register_width) {
        const __m256i input_chunk = _mm256_load_si256(input + i);
        // 找出哪些值大于 0 并在 nnz 中设置相应的位
        uint32_t nnz =
            _mm256_movemask_epi8(
                _mm256_cmpgt_epi8(input_chunk, _mm256_setzero_si256())
            );

        // 提取 nnz 中设置的位的索引
        while (nnz) {
            const int lsb_index = lsb(nnz);
            nnz &= nnz - 1; // 重置 nnz 中最低有效的设置位
            nnz_input_indices[num_nnz_input_indices++] = i + lsb_index;
        }
    }

    // 首先我们只复制偏差项。编译器擅长矢量化这个操作。
    // 也可以使用 memcpy
    for (int i = 0; i < layer.num_outputs; ++i) {
        output[i] = layer.biases[i];
    }

    const int num_chunks = layer.num_outputs / output_chunk_size;
    int i = 0;
    for (; i + 1 < num_nnz_input_indices; i += 2) {
        // 我们尽可能一次处理 2 个，因为我们可以更好地利用可用的内置函数。
        // 在在可视化中会更明显。
        const int input_id0 = nnz_input_indices[i+0];
        const int input_id1 = nnz_input_indices[i+1];
        const __m256i factor = _mm256_set1_epi32(
            input[input_id0] | (input[input_id1] << 16)
        );

        for (int j = 0; j < num_chunks; ++j) {
            const int output_offset0 = (j*2 + 0)*output_register_width;
            const int output_offset1 = (j*2 + 1)*output_register_width;

            // 权重的打包密度是输出的 2 倍。
            const int weight_offset  = (j*1 + 0)*output_register_width;

            // 每个块需要一次加载和存储。
            // 然而，如果输出小到足以展开并且
            // 所有输出可能都适合在寄存器中。
            // 虽然编译器可能不被允许自己这么做。
            __m256i sum0 = _mm256_load_si256(output + output_offset0);
            __m256i sum1 = _mm256_load_si256(output + output_offset1);

            // 记住，这里的权重是 16 位的，所以一个 __m256i 可以容纳 16 个。
            const __m256i col0 = _mm256_load_si256(
                layer.weights + input_id0 * layer.num_outputs + weight_offset
            );
            const __m256i col1 = _mm256_load_si256(
                layer.weights + input_id1 * layer.num_outputs + weight_offset
            );

            // 见下面的可视化
            m256_process_chunk(sum0, sum1, col0, col1, factor);

            _mm256_store_si256(output + output_offset0, sum0);
            _mm256_store_si256(output + output_offset1, sum1);
        }
    }

    // 处理剩余的单个输入
    for (; i < num_nnz_input_indices; ++i) {
        const int input_id = nnz_input_indices[i];
        const __m256i factor = _mm256_set1_epi32(input[input_id]);

        for (int j = 0; j < num_chunks; ++j) {
            const int output_offset0 = (j*2 + 0)*output_register_width;
            const int output_offset1 = (j*2 + 1)*output_register_width;

            const int weight_offset  = (j*1 + 0)*output_register_width;

            __m256i sum0 = _mm256_load_si256(output + output_offset0);
            __m256i sum1 = _mm256_load_si256(output + output_offset1);

            const __m256i col0 = _mm256_load_si256(
                layer.weights + input_id * layer.num_outputs + weight_offset
            );

            m256_process_chunk(sum0, sum1, col0, _mm256_setzero_si256(), factor);

            _mm256_store_si256(output + output_offset0, sum0);
            _mm256_store_si256(output + output_offset1, sum1);
        }
    }
    
    for (int j = 0; j < layer.num_outputs; j += output_register_width) {
        _mm256_store_si256(output + j, _mm256_srai_epi32(_mm256_load_si256(output + j), log2_weight_scale));
    }

    return output + layer.num_outputs;
}
```

##### m256_process_chunk

此函数采用 int16 权重，一个因子是作为 int32 广播的 2 个 int8 输入的组合，并产生 int32 输出。

![](img/m256_process_chunk.svg)

```cpp
inline void m256_process_chunk(__m256i& sum0, __m256i& sum1, __m256i col0, __m256i col1, __m256i factor) {
	// 我们交错两列，因为 madd 添加了相邻的值。
    // 这样我们就可以有效地将两列的结果相加。
    sum0 = _mm256_add_epi32(
        sum0, _mm256_madd_epi16(factor, _mm256_unpacklo_epi16(col0, col1))
    );
    sum1 = _mm256_add_epi32(
        sum1, _mm256_madd_epi16(factor, _mm256_unpackhi_epi16(col0, col1))
    );
}
```

#### 具有稀疏输入的线性层，替代方法

在第一种方法中，我们使用了 16 位权重，但也可以使用 8 位权重，解包的乐趣会稍微多一些。 我们还将看到另一种使用查找表计算非零输入索引的方法。 有关后者的更多方法和测量，请参阅 [此处](https://github.com/syzygy1/Cfish/issues/204#issue-944790893)。

```cpp
// 此实现需要更改布局并将权重扩展为 int16。
// 我们将转置权重，因为现在我们将遍历列而不是行。
void load_weights(
    const LinearLayer& layer,
    const int8_t* data
) {
    static_assert(is_same_v<LinearLayer::WeightType, int8_t>,
        "This approach requires weights to be 8 bit.");

    for (int i = 0; i < layer.num_outputs; ++i) {
        for (int j = 0; j < layer.num_inputs; ++j) {
            layer.weights[j*layer.num_outputs + i] = data[i*layer.num_inputs + j];
        }
    }

	// 现在不需要巧妙的技巧来调整权重。
    // 但是，我们还需要一个零权重列。 我们假设分配了足够的空间。
    for (int i = 0; i < layer.num_outputs; ++i) {
        layer.weights[layer.num_inputs*layer.num_outputs + i] = 0;
    }
}

// 最低有效位计算的 constexpr 版本。
static constexpr int lsb_constexpr(std::uint32_t v)
{
    int c = 0;
    if (!v) return 32;
    while (!(v & 1))
    {
        v >>= 1;
        ++c;
    }
    return c;
}

// 输入中非零位索引的查找表。
// std::array<std::uint16_t, 8> 的每个条目都可以解释为 __m128i。
alignas(64) static constexpr std::array<std::array<std::uint16_t, 8>, 256> LookupTableIndices = [](){
    std::array<std::array<std::uint16_t, 8>, 256> v{};
    for (int i = 0; i < 256; ++i)
    {
        int j = i;
        int k = 0;
        while(j)
        {
            const IndexType lsbIndex = lsb_constexpr(std::uint32_t(j));
            j &= j - 1;
            v[i][k] = lsbIndex;
            ++k;
        }
    }
    return v;
}();

// 字节弹出计数的查找表。
// 使用专用的 popcnt 指令可能会也可能不会更好。
static constexpr std::array<std::uint8_t, 256> LookupTableCounts = [](){
    std::array<std::uint8_t, 256> v{};
    for (int i = 0; i < 256; ++i)
    {
        int j = i;
        int k = 0;
        while(j)
        {
            j &= j - 1;
            ++k;
        }
        v[i] = k;
    }
    return v;
}();

int32_t* linear_sparse_input(
    const LinearLayer& layer,
    int32_t*           output,
    const int8_t*      input
) {
    // 我们将采用寄存器中累加器的平铺方法。
    // 类似于特征转换器的最佳实现方式。
    constexpr int input_register_width = 256 / 8;
    constexpr int chunk_size = 256 / 32;
    constexpr int num_chunks_per_tile = 8;
    constexpr int tile_size = chunk_size * num_chunks_per_tile;
    assert(layer.num_outputs % tile_size == 0, "We're processing 64 output elements at a time. Though it's easy to change it.");
    assert(num_chunks_per_tile % 4 == 0, "We're processing 4 chunks at a time.");
    constexpr int num_tiles = layer.num_outputs / tile_size;

	// 我们需要找出非零输入值的索引
    // 我们将使用查找表方法。 过度分配 16 个元素
    // 这样商店总是有效的（我们将使用更大的商店）
    uint16_t nnz_input_indices[layer.num_inputs + 16];
    int num_nnz_input_indices = 0;

    {
		// 这些将用于偏移查找的索引。
        // int16 查找的变体也是可能的（参见上面的链接）
        // 单独运行速度更快，但需要更多内存并且可能会破坏缓存。
        __m128i base = _mm_set1_epi16(0);
        __m128i increment = _mm_set1_epi16(8);
        for (int i = 0; i < layer.num_inputs; i += input_register_width) {
            const __m256i input_chunk = _mm256_load_si256(input + i);
            unsigned nnz = _mm256_movemask_epi8(_mm256_cmpgt_epi8(input_chunk, _mm256_setzero_si256()));

            unsigned b0 = (nnz) & 0xFF;
            unsigned b1 = (nnz >> 8) & 0xFF;
            unsigned b2 = (nnz >> 16) & 0xFF;
            unsigned b3 = (nnz >> 24) & 0xFF;

            unsigned c0 = LookupTableCounts[b0];
            unsigned c1 = LookupTableCounts[b1];
            unsigned c2 = LookupTableCounts[b2];
            unsigned c3 = LookupTableCounts[b3];

			// 在极端情况下这些存储可以到达layer.num_inputs之上。 这就是我们预分配的原因。
            // 只有第一个 c0 值很重要。
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices),
                _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b0])), base)
            );
            num_nnz_input_indices += c0;
            base = _mm_add_epi32(base, increment);

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices),
                _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b1])), base)
            );
            num_nnz_input_indices += c1;
            base = _mm_add_epi32(base, increment);

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices),
                _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b2])), base)
            );
            num_nnz_input_indices += c2;
            base = _mm_add_epi32(base, increment);

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(nnz_input_indices + num_nnz_input_indices),
                _mm_add_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b3])), base)
            );
            num_nnz_input_indices += c3;
            base = _mm_add_epi32(base, increment);
        }
    }

    // 我们将一次处理 4 个输入，并避免出现两个类似的循环
    // 我们将输入索引填充为 4 的倍数。对于添加的索引，我们使用虚拟输入
    // 将所有权重设置为 0。 
    while (num_nnz_input_indices % 4 != 0)
      nnz_input_indices[num_nnz_input_indices++] = layer.num_inputs;

    // 希望能够放入寄存器文件中。
    __m256i acc[num_chunks_per_tile];

    for (int i = 0; i < num_tiles; ++i)
    {
        const __m256i* biases_tile = reinterpret_cast<const __m256i*>(&layer.biases[i * tile_size]);
              __m256i* output_tile = reinterpret_cast<      __m256i*>(&      output[i * tile_size]);

        for (int k = 0; k < num_chunks_per_tile; ++k)
            acc[k] = _mm256_setzero_si256();

        for (int j = 0; j < num_nnz_input_indices; j += 4)
        {
            const __m256i  mul0 = _mm256_set1_epi16(input[nnz_input_indices[j+0]] | (input[nnz_input_indices[j+1]] << 8));
            const __m256i  mul2 = _mm256_set1_epi16(input[nnz_input_indices[j+2]] | (input[nnz_input_indices[j+3]] << 8));
            const __m256i* col0 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+0] * layer.num_outputs + i * tile_size]);
            const __m256i* col1 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+1] * layer.num_outputs + i * tile_size]);
            const __m256i* col2 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+2] * layer.num_outputs + i * tile_size]);
            const __m256i* col3 = reinterpret_cast<const __m256i*>(&layer.weights[nnz_input_indices[j+3] * layer.num_outputs + i * tile_size]);
            for (int k = 0; k < num_chunks_per_tile / 4; ++k)
            {
                // 由于 AVX2 将 256 位寄存器解释为 2 128 位寄存器，因此解包
                // 洗牌车道。 在获得最终结果时，我们必须考虑到这一点。
                m256_process_chunk_alternative(
                    acc[k*4 + 0], acc[k*4 + 1], acc[k*4 + 2], acc[k*4 + 3],
                         col0[k],      col1[k],      col2[k],      col3[k],
                            mul0,                       mul2
                );
            }
        }

        for (int k = 0; k < num_chunks_per_tile / 4; ++k)
        {
            // 我们必须重新调整车道。 查看可视化以获得更好的图片。
            const __m128i acc00 = _mm256_extracti128_si256(acc[k*4 + 0], 0);
            const __m128i acc01 = _mm256_extracti128_si256(acc[k*4 + 0], 1);
            const __m128i acc10 = _mm256_extracti128_si256(acc[k*4 + 1], 0);
            const __m128i acc11 = _mm256_extracti128_si256(acc[k*4 + 1], 1);
            const __m128i acc20 = _mm256_extracti128_si256(acc[k*4 + 2], 0);
            const __m128i acc21 = _mm256_extracti128_si256(acc[k*4 + 2], 1);
            const __m128i acc30 = _mm256_extracti128_si256(acc[k*4 + 3], 0);
            const __m128i acc31 = _mm256_extracti128_si256(acc[k*4 + 3], 1);

            output_tile[k*4 + 0] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_setr_m128i(acc00, acc10), biases_tile[k*4 + 0]), log2_weight_scale);
            output_tile[k*4 + 1] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_setr_m128i(acc20, acc30), biases_tile[k*4 + 1]), log2_weight_scale);
            output_tile[k*4 + 2] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_setr_m128i(acc01, acc11), biases_tile[k*4 + 2]), log2_weight_scale);
            output_tile[k*4 + 3] = _mm256_srai_epi32(_mm256_add_epi32(_mm256_setr_m128i(acc21, acc31), biases_tile[k*4 + 3]), log2_weight_scale);
        }
    }

    return output + layer.num_outputs;
}
```

##### m256_process_chunk_alternative

此函数采用对应于 4 个输入的 int8 权重，2 个因子是作为 int16 广播的 4 个 int8 输入的组合，并产生 int32 输出。

![](img/m256_process_chunk_alternative.svg)

```cpp
inline void m256_process_chunk_alternative(
    __m256i& acc0, __m256i& acc1, __m256i& acc2, __m256i& acc3,
    __m256i  col0, __m256i  col1, __m256i  col2, __m256i  col3,
    __m256i  mul0,                __m256i  mul2
) {
    // For madd.
    const __m256i ones = _mm256_set1_epi16(1);

    const __m256i prod0 = _mm256_maddubs_epi16(mul0, _mm256_unpacklo_epi8(col0, col1));
    const __m256i prod1 = _mm256_maddubs_epi16(mul0, _mm256_unpackhi_epi8(col0, col1));
    const __m256i prod2 = _mm256_maddubs_epi16(mul2, _mm256_unpacklo_epi8(col2, col3));
    const __m256i prod3 = _mm256_maddubs_epi16(mul2, _mm256_unpackhi_epi8(col2, col3));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(ones, _mm256_unpacklo_epi16(prod0, prod2)));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(ones, _mm256_unpackhi_epi16(prod0, prod2)));
    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(ones, _mm256_unpacklo_epi16(prod1, prod3)));
    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(ones, _mm256_unpackhi_epi16(prod1, prod3)));
}
```

#### 带有稀疏输入和块状稀疏输出的线性层

让我们进一步深入。到目前为止，所有的线性层都有密集的输出，但我们可以考虑一个层次，其中每个输入只连接到输出的一个子集。我们可以认为，在没有连接存在的地方，权重为0。为了使得能够有效地实施并考虑向量化，我们必须将整个权重块都置为0。例如，一个16x128的权重矩阵，每个输入有2个非零的1x16块，可能是这样的：

![](img/m256_block_sparse_weight_matrix.svg)

对于AVX2，这样的块至少需要8个int32s（输出值的类型）宽，但我们只考虑16宽的块，因为这更方便。采用这种方法，人们可以例如有一个有256个输出的线性层，但每个输入只有4个（这个常数对于能够编写优化的代码非常重要）非零权重块，大小为16，实际上，每个输入只影响64个输出。

为了支持它，正向传递中有一些额外的工作负担，并且它没有像之前的情况那样很好地向量化，但对于某些架构来说，这可能仍然是一个胜利。

然而，采用这种方法，训练需要意识到这一点，并尝试创建那些0权重的块，而不会过多地损害网络。这可以通过权重修剪来实现，这将在后面详述。推断代码将与稀疏输入的线性层非常相似。


```cpp
void load_weights(
    const LinearLayer& layer,
    const int8_t* data
) {
		// 不过，这与稀疏输入的情况相同
		// 权重矩阵不再连续，我们需要填充
		// 一些块索引来了解哪些权重对应于哪些输出。
        // 这可以通过在加载期间发现零块来完成，
        // 或者使用不同的序列化格式并预先计算块索引。
        // 我们将在这里省略这一点并假设layer.nnz_block_ids[input_id][4]
        // 包含与每个输入对应的非零权重块索引。
}

int32_t* linear_sparse_input_block_sparse_output(
    const LinearLayer& layer,
    int32_t*           output,
    const int8_t*      input
) {
    static_assert(is_same_v<LinearLayer::WeightType, int16_t>,
        "This approach requires weights to be 16 bit. Otherwise it's hard to widen the multiplication output to 32 bits.");

    constexpr int register_width = 256 / 8;
    constexpr int input_register_width = register_width; // uint8_t
    constexpr int output_register_width = register_width / 4; // int32_t
    constexpr int output_chunk_size = output_register_width * 2; // we will be processing 2 registers at a time
    assert(layer.num_outputs % output_chunk_size == 0, "We're processing 16 output elements at a time");
    assert(layer.num_inputs % input_register_width == 0);

    uint16_t nnz_input_indices[layer.num_inputs];
    int num_nnz_input_indices = 0;

    for (int i = 0; i < layer.num_inputs; i += input_register_width) {
        const __m256i input_chunk = _mm256_load_si256(input + i);
        uint32_t nnz =
            _mm256_movemask_epi8(
                _mm256_cmpgt_epi8(input_chunk, _mm256_setzero_si256())
            );

        while (nnz) {
            const int lsb_index = lsb(nnz);
            nnz &= nnz - 1; // reset the least significant set bit in nnz
            nnz_input_indices[num_nnz_input_indices++] = i + lsb_index;
        }
    }

    for (int i = 0; i < layer.num_outputs; ++i) {
        output[i] = layer.biases[i];
    }

    const int num_chunks = layer.num_outputs / output_chunk_size;
    // 总是需要权衡。 我们无法同时处理两个输入，因为
    // 他们可能有不同的非零权重块。 使其明显变慢。
    // AVX512 可能有一些技巧，但 AVX2 对于此用例来说相当有限。
    for (int i = 0; i < num_nnz_input_indices; ++i) {
        const int input_id = nnz_input_indices[i]
        const __m256i factor = _mm256_set1_epi32(input[input_id]);

        // We have hardcoded 4 16-wide non-zero weight blocks per input.
        for (int j = 0; j < 4; ++j) {
            const int block_id = layer.nnz_block_ids[input_id][j];
            const int output_offset0 = (block_id*2 + 0)*output_register_width;
            const int output_offset1 = (block_id*2 + 1)*output_register_width;

            const int weight_offset  = (block_id*1 + 0)*output_register_width;

            __m256i sum0 = _mm256_load_si256(output + output_offset0);
            __m256i sum1 = _mm256_load_si256(output + output_offset1);

            const __m256i col0 = _mm256_load_si256(
                layer.weights + input_id * layer.num_outputs + weight_offset
            );

            m256_process_chunk(sum0, sum1, col0, _mm256_setzero_si256(), factor);

            _mm256_store_si256(output + output_offset0, sum0);
            _mm256_store_si256(output + output_offset1, sum1);
        }
    }
    
    for (int i = 0; i < layer.num_outputs; i += output_register_width) {
        _mm256_store_si256(output + i, _mm256_srai_epi32(_mm256_load_si256(output + i), log2_weight_scale));
    }

    return output + layer.num_outputs;
}
```

#### ClippedReLU

截断并不难，更复杂的部分是转换。我们还需要两个版本，一个是int16 -> int8，另一个是int32 -> int8。

##### int16 -> int8

![](img/crelu16.svg)

```cpp
int8_t* crelu16(,
          int      size,   // no need to have any layer structure, we just need the number of elements
          int8_t*  output, // the already allocated storage for the result
    const int16_t* input   // the input, which is the output of the previous linear layer
) {
    constexpr int in_register_width = 256 / 16;
    constexpr int out_register_width = 256 / 8;
    assert(size % out_register_width == 0, "We're processing 32 elements at a time");
    const int num_out_chunks = size / out_register_width;

    const __m256i zero    = _mm256_setzero_si256();
    const int     control = 0b11011000; // 3, 1, 2, 0; lane 0 is the rightmost one

    for (int i = 0; i < num_out_chunks; ++i) {
        const __m256i in0 = _mm256_load_si256(&input[(i * 2 + 0) * in_register_width]);
        const __m256i in1 = _mm256_load_si256(&input[(i * 2 + 1) * in_register_width]);

        const __m256i result =
            // packs changes the order, so we need to fix that with a permute
            _mm256_permute4x64_epi64(
                // clamp from below
                _mm256_max_epi8(
                    // packs saturates to 127, so we only need to clamp from below
                    _mm256_packs_epi16(in0, in1),
                    zero
                ),
                control
            );

        _mm256_store_si256(&output[i * out_register_width], result);
    }

    return output + size;
}
```

##### int32 -> int8

![](img/crelu32.svg)

```cpp
int8_t* crelu32(,
          int      size,   // no need to have any layer structure, we just need the number of elements
          int8_t*  output, // the already allocated storage for the result
    const int32_t* input   // the input, which is the output of the previous linear layer
) {
    constexpr int in_register_width = 256 / 32;
    constexpr int out_register_width = 256 / 8;
    assert(size % out_register_width == 0, "We're processing 32 elements at a time");
    const int num_out_chunks = size / out_register_width;

    const __m256i zero    = _mm256_setzero_si256();
    const __m256i control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    for (int i = 0; i < num_out_chunks; ++i) {
        const __m256i in0 =
            _mm256_packs_epi32(
                _mm256_load_si256(&input[(i * 4 + 0) * in_register_width]),
                _mm256_load_si256(&input[(i * 4 + 1) * in_register_width])
            );
        const __m256i in1 =
            _mm256_packs_epi32(
                _mm256_load_si256(&input[(i * 4 + 2) * in_register_width]),
                _mm256_load_si256(&input[(i * 4 + 3) * in_register_width])
            );

        const __m256i result =
            _mm256_permutevar8x32_epi32(
                _mm256_max_epi8(
                    _mm256_packs_epi16(in0, in1),
                    zero
                ),
                control
            );

        _mm256_store_si256(&output[i * out_register_width], result);
    }

    return output + size;
}
```

#### Quantmoid4

As previously mentioned, we will be considering the variant with output in range `[0, 126]`. Let's remind ourselves about the equation we need to implement.

![Quantmoid4 equation](img/quantmoid4_equation.png)

First thing to notice is that `min(x, y) - y` can be replaced by `-subs(y, x)`, where `subs` is unsigned subtraction with saturation, in this case saturation to the range `[0, 255]`.

For handling the "piece-wise" nature we can either use a copysign function, or a blend function. Since blending is available in AVX2 we will use this.

The code can then be as simple as the following:

```cpp
// 由于输出在 int8 范围外始终为 0/1，理论上输入可以为（饱和）int8，
// 但是 AVX2 没有 int8 乘法，所以我们会方便地使用 int16。
int8_t* quantmoid4(
          int      size,
    const int16_t* input,
          int8_t*  output
) {
    constexpr int in_register_width = 256 / 16;
    constexpr int out_register_width = 256 / 8;
    assert(size % out_register_width == 0); // 我们在这里不会处理剩余的部分
    const int num_out_chunks = size / out_register_width;

    const __m256i cst_127_epi16 = _mm256_set1_epi16(127);
    const __m256i cst_126_epi8 = _mm256_set1_epi8(126);

    // 由于 AVX2 处理在 128 位通道之间交错，我们将必须在最后还原，
    // 将两个处理过的输入组合成一个输出。
    // 此 Control 包含如何排列结果的 64 位通道的信息。Control = [0, 2, 1, 3]。
    constexpr int Control = 0b11011000;
    for (int i = 0; i < num_out_chunks; ++i)
    {
        __m256i v0 = _mm256_load_si256(&input[(i * 2 + 0) * in_register_width]);
        __m256i v1 = _mm256_load_si256(&input[(i * 2 + 1) * in_register_width]);

        // 我们将需要初始输入的符号用于后面的混合。
        // Blend 仅使用最高位进行选择，那恰好是符号位。
        __m256i sign = _mm256_packs_epi16(v0, v1);

        // 由之前给出的等式可得。
        // v0 = min(abs(input[i]), 127) - 127;
        v0 = _mm256_subs_epu16(cst_127_epi16, _mm256_abs_epi16(v0));
        v1 = _mm256_subs_epu16(cst_127_epi16, _mm256_abs_epi16(v1));

        // 由于我们稍后要使用 mulhi，我们必须准备输入，使得
        // 乘法后的高部分（16 个最高位，因为 16 位乘法产生一个
        // 32 位的结果）可以正确地放置。我们希望后面能通过除以
        // 256==2^8（右移 8 位）来操作，所以我们需要处理掉 8 个
        // 额外的位（来自完整的 32 位结果）。所以我们可以先向左
        // 移动 4 位，这个操作会在乘法（平方）后变为 8，所以 32 位
        // 结果的 16 位高部分会完全提取我们想要的部分。
        v0 = _mm256_slli_epi16(v0, 4);
        v1 = _mm256_slli_epi16(v1, 4);

        v0 = _mm256_mulhi_epi16(v0, v0);
        v1 = _mm256_mulhi_epi16(v1, v1);

        // 现在我们可以在这之后转换为 int8。
        v0 = _mm256_packs_epi16(v0, v1);

        // 根据输入符号，在 v 和 126-v 之间进行混合，从而我们有效地
        // 评估了分段函数的正确部分。
        v0 = _mm256_blendv_epi8(_mm256_subs_epi8(cst_126_epi8, v0), v0, sign);

        // 由于 AVX2 的语义，对输出进行反交错处理。
        _mm256_store_si256(&output[i * out_register_width], _mm256_permute4x64_epi64(v0, Control));
    }

    return output + size;
}
```

#### 池化层

##### 平均池化

具体的实现将取决于每个输出要采取多少输入。在这里，我们将展示一个实现，它将2个输入减少到1个输出，并将在uint8输入上工作。

注意，通常不最优的方式是取相邻的值。例如在这里，我们将输入一分为二并平均这两半，因为在AVX2中没有允许做得很好的指令。另外，要注意舍入差异 - 例如，AVX2平均值会舍入到最接近的整数，而简单的除法会舍入到0。

```cpp
void average_pooling_2(
          int      size,
    const uint8_t* input,
          uint8_t*  output
) {
    constexpr int register_width = 256 / 8;
    assert(size % (register_width * 2) == 0); // We won't bother with the remainder here
    const int num_out_chunks = size / (register_width * 2);

    for (int i = 0; i < num_out_chunks; ++i)
    {
        __m256i v0 = _mm256_load_si256(&input[ i                   * register_width]);
        __m256i v1 = _mm256_load_si256(&input[(i + num_out_chunks) * register_width]);

        _mm256_store_si256(&output[i * register_width], _mm256_avg_epu8(v0, v1));
    }
}
```

##### 最大池化

几乎是一样的。尽管在AVX2中，使用`max`的输入/输出类型有更多的选择。

```cpp
void max_pooling_2(
          int      size,
    const uint8_t* input,
          uint8_t*  output
) {
    constexpr int register_width = 256 / 8;
    assert(size % (register_width * 2) == 0); // We won't bother with the remainder here
    const int num_out_chunks = size / (register_width * 2);

    for (int i = 0; i < num_out_chunks; ++i)
    {
        __m256i v0 = _mm256_load_si256(&input[ i                   * register_width]);
        __m256i v1 = _mm256_load_si256(&input[(i + num_out_chunks) * register_width]);

        _mm256_store_si256(&output[i * register_width], _mm256_max_epu8(v0, v1));
    }
}
```

##### 乘积池化

这个更复杂，因为乘法引入了一个非线性，需要在量化中考虑。我们将考虑一个在输入范围`[0, 127]`上工作的版本，其中0对应于浮点数0.0，127对应于浮点数1.0。我们希望输出保持这个范围，但为了做到这一点，我们必须除以127（因为`127*127/127`将是`1.0*1.0=1.0`的期望结果）。实际上我们必须牺牲一些输出范围，而是除以128，实际上产生的输出范围在`[0, 126]`（1.0点仍然逻辑上可以是127，但输出不会覆盖整个`[0.0, 1.0]`范围）。这个变化必须在训练者中考虑到，因为我们实际上正在改变函数，不仅乘以输入，而且缩放输出。不管怎样，下面是用AVX2进行优化实现的C++代码可能是什么样子的：

（在Stockfish中，这是在特征变换器激活后应用的，因此是int16->uint8的例子）

```cpp
void product_pooling_2(
          int      size,
    const int16_t* input,
          uint8_t* output
) {
    constexpr int in_register_width = 256 / 16;
    constexpr int out_register_width = 256 / 8;
    assert(size % (out_register_width * 2) == 0); // We won't bother with the remainder here
    const int num_out_chunks = size / (out_register_width * 2);

    // For deinterleave
    constexpr int Control = 0b11011000;

    for (int i = 0; i < num_out_chunks; ++i)
    {
        // We process 4 input registers at a time and produce one output register.
        // This is because we do 2->1 reduction and input type is twice the width of the output.
        const __m256i v0a = _mm256_load_si256(&input[(i * 2 + 0)                  * in_register_width]);
        const __m256i v0b = _mm256_load_si256(&input[(i * 2 + 1)                  * in_register_width]);
        const __m256i v1a = _mm256_load_si256(&input[(i * 2 + 0 + num_out_chunks) * in_register_width]);
        const __m256i v1b = _mm256_load_si256(&input[(i * 2 + 1 + num_out_chunks) * in_register_width]);

        // Multiply and divide by 128 (right shift by 7), rounding towards 0.
        const __m256i pa = _mm256_srli_epi16(_mm256_mullo_epi16(v0a, v1a), 7);
        const __m256i pb = _mm256_srli_epi16(_mm256_mullo_epi16(v0b, v1b), 7);

        // Deinterleave
        out[j] = _mm256_permute4x64_epi64(_mm256_packs_epi16(pa, pb), Control);
    }
}
```

这一层也可以很好地与Clipped ReLU结合，但首先将输入限制在特定范围内（以融合的方式进行可以减少加载/存储的次数）。


```cpp
void max_pooling_2(
          int      size,
    const uint8_t* input,
          uint8_t*  output
) {
    constexpr int register_width = 256 / 8;
    assert(size % (register_width * 2) == 0); // We won't bother with the remainder here
    const int num_out_chunks = size / (register_width * 2);

    for (int i = 0; i < num_out_chunks; ++i)
    {
        __m256i v0 = _mm256_load_si256(&input[ i                   * register_width]);
        __m256i v1 = _mm256_load_si256(&input[(i + num_out_chunks) * register_width]);

        _mm256_store_si256(&output[i * register_width], _mm256_max_epu8(v0, v1));
    }
}
```

### 在训练器中考虑量化

#### 范围

添加（相当激进的）量化已经减少了权重和偏置的可能值范围。然而，我们可以忽略特征转换器和所有偏置，因为它们使用大的整数类型，我们不期望达到限制。问题的情况是线性层的int8权重，例如在Stockfish中，只能达到约2（激活范围在0..1）。这可能是一个大问题，因为训练可以通过超过四舍五入的方式偏离量化表示。为防止这种情况发生，有必要在训练器内部以某种方式限制这些参数的范围。到目前为止，最简单的方法是在每次优化步骤后修改优化器以将值限制在可用范围内。这些最小值和最大值可以被传递，例如在优化器中注册可优化参数时。

##### 在优化器内部

考虑这个问题的一种方式是直接在优化器中。这很好，因为剪裁是在步骤之后直接应用的，但需要访问优化器的源代码。例如：

```python
# The min/max constants are specific for the Stockfish quantization scheme.
train_params = [
    {'params' : [self.ft.weight, self.ft.bias] },
    {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.l1.bias] },
    {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.l2.bias] },
    {'params' : [self.output.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
    {'params' : [self.output.bias] },
]
optimizer = ranger.Ranger(train_params, lr=LR, betas=(.9, 0.999), eps=1.0e-7)
```

and then in the optimizer:

```python
class Ranger(Optimizer):
    def __init__([...]):
        [...]
        defaults = dict([...]
                        min_weight=None, max_weight=None)

def step(self, closure=None):
    [...]

    for group in self.param_groups:
        for p in group['params']:
            ...
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            if min_weight is not None and max_weight is not None:
                p_data_fp32.clamp_(min_weight, max_weight)
```

##### 在优化器之外

或者，为了更大的灵活性，可以在优化器之外进行：

```python
# The min/max constants are specific for the Stockfish quantization scheme.
self.weight_clipping = [
    {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.output.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
]
```

```python
# and call this in some step function
def _clip_weights(self):
    for group in self.weight_clipping:
        for p in group['params']:
            p_data_fp32 = p.data
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)
```

##### 考虑虚拟层（因子化）

有时候更复杂的架构会在训练期间使一些层的参数成为两层的和。就像特征因子化，但对整个层（例如[这个](https://chat.openai.com/?model=gpt-4#multiple-psqt-outputs-and-multiple-subnetworks)）。我们可以这样考虑：

```python
# The min/max constants are specific for the Stockfish quantization scheme.
self.weight_clipping = [
    {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64, 'virtual_params' : self.some_virtual_factor.weight },
    {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.output.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
]
```

```python
def _clip_weights(self):
    for group in self.weight_clipping:
        for p in group['params']:
            p_data_fp32 = p.data
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            if 'virtual_params' in group:
                virtual_params = group['virtual_params']
                # The virtual layer is usually N times smaller
                xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                expanded_virtual_layer = virtual_params.repeat(xs, ys)
                min_weight_t = p_data_fp32.new_full(p_data_fp32.shape, min_weight) - expanded_virtual_layer
                max_weight_t = p_data_fp32.new_full(p_data_fp32.shape, max_weight) - expanded_virtual_layer
                p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
            else:
                p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)
```

#### 非可微层

你可能想要使用一些从一开始就为量化推理设计的非可微层。Quantmoid4就是这样的一层。在这种情况下，我们有两个选项：

1. 在浮点域中直接使用该函数。
2. 使用其他函数。

第一个选项是最简单的，但可能仍然在边界点上出现问题。例如，Quantmoid4使用非平滑函数，这将导致梯度消失。这是个问题。另外，这通常是很多简单的操作，可能会拖慢训练器的速度（通过扩大反向传播图）。

第二个选项非常有吸引力，特别是因为通常量化函数是对其他某种平滑函数的近似，在这种情况下是`sigmoid(4x)`。在某些情况下，只需要替换梯度，留下量化版本进行前向传递就足够了。

##### 为训练安全的合并Quantmoid4定制内核

Quantmoid4是一个足够好的近似，我们可以直接用它进行前向传递，并使用`sigmoid(4x)`梯度进行反向传递！

```python
import torch
import cupy as cp

# For lazy compilation.
quantmoid_kernels = dict()
def make_quantmoid4_forward_kernel():
    key = 'quantmoid4_forward_kernel'
    if not key in quantmoid_kernels:
        # an approximation of sigmoid(x*4)
        quantmoid4_forward_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void quantmoid4_forward(
                const float*   const input,
                      float*   const output,
                const int            total
            ) {
                const int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= total)
                   return;

                // Remember that we want to scale the output to [0.0, 1.0]
                const float x = input[i];
                const float v = min(floor(abs(x * 127.0f)), 127.0f) - 127.0f;
                const float vv = floor(v * v / 256.0f);
                const float vvv = x > 0.0f ? 126.0f - vv : vv;
                output[i] = vvv / 127.0f;
            }
        ''',
            'quantmoid4_forward'
        )
        quantmoid4_forward_kernel.compile()
        quantmoid_kernels[key] = quantmoid4_forward_kernel
    return quantmoid_kernels[key]

# Now we define a python function that will encapsulate that raw kernel.
def _quantmoid4(x):
    assert x.is_contiguous()
    assert x.is_cuda
    assert x.dtype == torch.float32

    kernel = make_quantmoid4_forward_kernel()
    device = x.device
    count = x.numel()
    output = torch.empty(*x.shape, dtype=torch.float32, device=device, requires_grad=False)

    kernel(
        grid=((count + 1023) // 1024,),
        block=(1024,),
        args=(
            x.data_ptr(),
            output.data_ptr(),
            count
        )
    )
    return output

# And now we define a class that pytorch will understand.
# Here we substitute the gradient by the gradient for sigmoid(4x).
class Quantmoid4(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(torch.sigmoid(input * 4.0))
    return _quantmoid4(input)

  @staticmethod
  def backward(ctx, grad_output):
    sigmoid_output, = ctx.saved_tensors
    return grad_output * (1.0 - sigmoid_output) * 4.0 * sigmoid_output

quantmoid4 = Quantmoid4.apply

# Output for some verification
if __name__ == '__main__':
    for i in range(-255, 255):
        x = i / 127.0
        print(x, quantmoid4(torch.tensor([x]).cuda()), quantmoid4(torch.tensor([x]).cuda())-torch.sigmoid(torch.tensor([x]).cuda()*4.0))
```

## 优化训练器（CUDA）

### 使用自定义CUDA内核

如何运行我们自己的内核？我们不需要带有CUDA编译器的复杂设置吗？CuPy来解救。CuPy是一个Python库，它允许使用包含CUDA代码的普通Python字符串轻松创建CUDA内核。CuPy为我们处理编译和所有其他事情。例如：

```python
import cupy as cp

# 创建内核
kernel = cp.RawKernel(r'''
void kernel_name(...) {
    // 你通常的内核代码
}
''', 'kernel_name')

# 可选择编译，否则将在首次使用时编译
kernel.compile()

# 运行内核
kernel(
    grid=(batch_size,), # 网格形状
    block=(num_threads,), # 块形状
    args=(...) # 传递给内核的参数
)
```

PyTorch张量可以通过使用`.data_ptr()`轻松地传递给内核，该方法会返回指向张量的指针。然而，必须确保内存是连续的。

### 特征转换器

到目前为止，我们一直在使用pytorch的稀疏矩阵乘法进行特征转换，但它们的实现并不好，我们有额外的假设可以使用。

1. 我们对每个位置的非零元素数量有一个上限。
2. 我们有大批量

因此，我们可以将特征索引从一个形状为`[total_num_active_features, 2]`的2D张量替换为一个形状为`[batch_size, max_num_features]`的2D张量，该张量对于每个值包含一个特征索引，位置索引是已知的。我们现在需要以某种方式处理特征数量小于`max_num_features`的位置。一种方法是为未使用的位置分配一个特征索引`-1`，然后在内核中忽略这样的索引。

这种方法显然也需要修改数据加载器，但现在它会变得更简单，因为我们不再需要解决pytorch中问题多多的稀疏张量。让我们先看看修改后的数据加载器。

#### New data loader

```cpp
struct SparseBatch {
    SparseBatch(const std::vector<TrainingDataEntry>& entries) {
        size = entries.size();

        max_active_features = MAX_ACTIVE_FEATURES;

        stm = new float[size];
        score = new float[size];

		// 索引的新布局，现在是 [size][MAX_ACTIVE_FEATURES]。
        // 另外，我们不需要对索引进行排序，因为新的实现
        // 无论顺序如何，速度都很快！
        white_features_indices = new int[size * MAX_ACTIVE_FEATURES];
        black_features_indices = new int[size * MAX_ACTIVE_FEATURES];

        fill(entries);
    }

    void fill(const std::vector<TrainingDataEntry>& entries) {
        ...
    }

    int size;
    int max_active_features;

    float* stm;
    float* score;
    int* white_features_indices;
    int* black_features_indices;

    ~SparseBatch()
    {
        delete[] stm;
        delete[] score;
        delete[] white_features_indices;
        delete[] black_features_indices;
    }
};
```

and in python

```python
# SparseBatch类定义
class SparseBatch(ctypes.Structure):
    # 定义结构体的字段
    _fields_ = [
        ('size', ctypes.c_int),  # 大小
        ('max_active_features', ctypes.c_int),  # 最大活跃特征数量
        ('stm', ctypes.POINTER(ctypes.c_float)),  # stm指针
        ('score', ctypes.POINTER(ctypes.c_float)),  # 分数指针
        ('white_features_indices', ctypes.POINTER(ctypes.c_int)),  # 白色特征索引
        ('black_features_indices', ctypes.POINTER(ctypes.c_int))  # 黑色特征索引
    ]

    # 获取tensor方法
    def get_tensors(self, device):
        # 这仅仅是说明性的。在现实中，你可能需要将它们传输到GPU。
        # 你也可以异步地完成这个操作，但要确保源存活足够长的时间以完成复制。

        stm_t = torch.from_numpy(np.ctypeslib.as_array(self.stm, shape=(self.size, 1)))
        score_t = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1)))

        # 现在我们不必烦恼稀疏的PyTorch张量！
        # 也不需要进行转置，因为我们可以控制布局！
        white_features_indices_t = torch.from_numpy(
            np.ctypeslib.as_array(self.white_features_indices, shape=(self.size, self.max_active_features)))
        black_features_indices_t = torch.from_numpy(
            np.ctypeslib.as_array(self.black_features_indices, shape=(self.size, self.max_active_features)))

        # 值都是1，所以我们可以轻松地在原地创建这些张量。
        # 无需经过复制。
        white_features_values_t = torch.ones(self.num_active_white_features)
        black_features_values_t = torch.ones(self.num_active_black_features)

        # 不再需要合并！无论输入是否排序，我们的实现都将快速！
        return white_features_indices_t, white_features_values_t, black_features_indices_t, black_features_values_t, stm_t, score_t

# 让ctypes知道如何理解这种类型。
SparseBatchPtr = ctypes.POINTER(SparseBatch)
```

#### 特征转换器前向内核

现在，让我们尝试编写一个用于特征转换器的自定义CUDA内核。在这一点上，你应该对特征转换器的工作方式以及如何在C中实现它有很好的理解。在这里，我们将需要两个内核，一个用于前向传递，一个用于后向传递。我们将以通用的方式编写这些内核，这些内核使用值，但是对于一些用途，当然可以假定所有的值都是1（取决于特征集和因子化方案）。在注释中添加注释来展示内核将是最容易的：

```c++
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @假设:
        块的维度必须为 (BATCH_SIZE,)
        线程的维度必须为 (N,), 其中
        N * output_thread_slice_size == output_size.

    @参数: feature_indices
        一个形状为 (BATCH_SIZE, max_active_features) 的矩阵
        它包含批次中每个位置的活跃特征的索引。
        特征索引为 -1 表示该插槽为空
        并且不会为其累积权重。更进一步
        这个块的其它索引也将不会被考虑。
        这些索引形成了一个隐式的形状为
        (BATCH_SIZE, NUM_INPUTS) 的矩阵，其中第一维度索引是
        通过内存位置 (BATCH_SIZE) 推断出来的，而
        第二维度索引存储在 feature_indices 矩阵中。
        feature indices 的类型为 int32_t.

    @参数: feature_values
        一个形状为 (BATCH_SIZE, max_active_features) 的矩阵
        它包含在 feature_indices 中对应的特征索引的值 (arity)。
        feature value (arity) 的类型为 float32.

    @参数: weight
        形状为 (NUM_INPUTS, output_size) 的权重矩阵。
        权重必须是 float32 类型。

    @参数: bias
        形状为 (output_size,) 的偏置向量。
        偏置值必须是 float32 类型。

    @参数: output
        形状为 (BATCH_SIZE, output_size) 的输出矩阵。
        它可能尚未初始化，偏置总是首先复制
        到输出中。
        输出值必须是 float32 类型。

    @常量: max_active_features
        单个位置的活跃特征的最大数量。这个值决定了
        输入的形状。
        此值的类型为 uint32_t。

    @常量: output_size
        输出的数量。必须与权重和偏置的形状匹配。
        此值的类型为 uint32。

    @常量: output_thread_slice_size
        每个线程需要处理的输出数量。必须为 output_size/num_threads。
        相当于 output_size/threadDim.x，但是在运行时计算它是浪费的。
        此值的类型为 uint32。
*/

void feature_transformer_slice_forward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
    const float*   const bias,
          float*   const output
) {
    // 这个想法是每个 CUDA 块处理一个位置，每个块中
    // 有 N 个线程，每个线程都在输出的一部分上工作。

    // 这些值是常数以允许更多的优化。
    // 由于我们使用 CuPy 免费提供 JIT 编译，这些
    // 值可以例如通过字符串插值设置
    // 每当需要特定参数化的内核时。
    const uint32_t       max_active_features      = ...;
    const uint32_t       output_thread_slice_size = ...;
    const uint32_t       output_size              = ...;

    // 我们获取一些所有线程共享的内存。
    // 理论上我们不在线程之间访问它，所以这可以
    // 是本地的，但是没有 __shared__ 定义的数组
    // 放在全局内存中可能会慢，而且
    // 我们必须依赖编译器对它进行优化。
    __shared__
          float          shared_output[output_size];

    // 1 个块是 1 个位置
    const uint32_t       block_idx           = blockIdx.x;
    // 每个线程只处理位置的一小部分输出。
    const uint32_t       slice_offset        = threadIdx.x * output_thread_slice_size;

    // 每个线程只填充位置的一小部分输出。
    // 这里我们计算进入输出 [batch_size, output_size] 数组的偏移
    // 在这里我们需要将这个线程的结果放置的地方。
          float*   const output_slice        = output + block_idx * output_size + slice_offset;
    // 以及其他类似的东西。
    const float*   const bias_slice          = bias                             + slice_offset;
          float*         shared_output_slice = shared_output                    + slice_offset;

    // 当我们使用 pytorch 的稀疏矩阵时，我们需要为每个值放置 2 个索引，
    // 他们是位置索引和特征索引。现在我们正在利用
    // 我们的第一个假设 - 我们有一个形状为 [batch_size, max_active_features] 的稠密矩阵，
    // 我们只存储每个特征的一个索引，位置索引是已知的。
    const int32_t* const feature_index_row   = feature_indices + block_idx * max_active_features;
    const float*   const feature_value_row   = feature_values  + block_idx * max_active_features;

    #pragma unroll
    // 将偏置复制到 "本地" 内存。
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        shared_output_slice[s] = bias_slice[s];
    }

    // 每个线程都遍历所有活跃的特征。
    for (uint32_t k = 0; k < max_active_features; ++k)
    {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        // 我们让特征索引为 -1 停止执行。
        // 对于所有线程，这个条件是一样的，所以我们可以早点退出
        // 并获得一些额外的性能。
        如果 feature_index 不等于 -1
        {
            // 计算我们需要累积哪些权重。
            const float* const weight_slice = weight + feature_index * output_size + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < output_thread_slice_size; ++s)
            {
                // 并将权重累积到 "本地" 内存。
                shared_output_slice[s] += weight_slice[s] * feature_value;
            }
        } else break;
    }

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        // 只有在最后我们才将结果放回全局内存。
        output_slice[s] = shared_output_slice[s];
    }
}
```

#### F特征转换器后内核

```Cuda
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: weight_grad
        The weight gradient matrix of shape (NUM_INPUTS, output_size).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Weights must be of type float32.

    @param: bias_grad
        The bias gradient vector of shape (output_size,).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Bias values must be of type float32.

    @param: output_grad
        An output gradient matrix of shape (BATCH_SIZE, output_size).
        Output values must have type float32.
*/
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad
) {{
    // The helper indices and pointers we compute are very similar
    // to the forward pass, we're just going to be doing it backwards.
    const uint32_t       max_active_features      = ...;
    const uint32_t       output_thread_slice_size = ...;
    const uint32_t       output_size              = ...;

    // We don't really need to store this in the shared memory, because
    // it's almost surely cached, but since it's free and we do
    // use it many times in this kernel we might as well do it.
    __shared__
          float          shared_output_grad[output_size];

    const uint32_t       block_idx                = blockIdx.x;
    const uint32_t       slice_offset             = threadIdx.x * output_thread_slice_size;

    const float*   const output_grad_slice        = output_grad + block_idx * output_size + slice_offset;
          float*   const bias_grad_slice          = bias_grad                             + slice_offset;
          float*         shared_output_grad_slice = shared_output_grad                    + slice_offset;

    const int32_t* const feature_index_row        = feature_indices + block_idx * max_active_features;
    const float*   const feature_value_row        = feature_values  + block_idx * max_active_features;

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        // Copy the values to "local" memory to hopefully speed up the repeated access.
        shared_output_grad_slice[s] = output_grad_slice[s];
    }

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        // x*w+b=y, so the bias gradient is just increased by the output gradient.
        const float sog = shared_output_grad_slice[s];
        // We expect this layer to come before a ClippedReLU so there will be a lot of zeros.
        // Also our kernel is completely memory bound, so we can utilize this to remove
        // redundant additions.
        if (sog != 0.0f)
        {
            // Due to how Nvidia GPUs work, since Kepler architecture, atomic
            // additions execute in specialized units that are closer to global memory.
            // Our access is mostly random, so be benefit here two-fold:
            // 1. atomicAdd executes **faster** than += because it's closer to memory
            // 2. we "rarely" have two atomic accesses to the same memory location.
            // We have to use atomic additions either way, because we're modifying
            // one gradient matrix (instead of multiple outputs as in the forward case),
            // so this is fortunate for us.
            atomicAdd(&bias_grad_slice[s], sog);
        }
    }

    // Same loop as in forward, but we accumulate the gradients now.
    for (uint32_t k = 0; k < max_active_features; ++k)
    {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        // Exit early after all active indices are processed.
        if (feature_index != -1)
        {
            float* const weight_grad_slice = weight_grad + feature_index * output_size + slice_offset;
            #pragma unroll
            for (int s = 0; s < output_thread_slice_size; ++s)
            {
                const float sog = shared_output_grad_slice[s];
                // Same optimization as in the case of the bias.
                if (sog != 0.0f)
                {
                    // x*w+b=y, so we accumulate output gradient multiplied by x (input).
                    atomicAdd(&weight_grad_slice[s], sog * feature_value);
                }
            }
        } else break;
    }
}
```

#### FeatureTransformerSlice 层

现在我们创建一个安全的包装器，以便可以轻松地从 PyTorch 中使用内核。

```python
class FeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        # Save the required stuff for the backward pass.
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        # A lot of assertions are needed to ensure the correctness.
        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        # Implementation for make_feature_transformer_slice_forward_kernel not provided. It could
        # for example dynamically create and cache the kernels.
        kernel, num_threads = make_feature_transformer_slice_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,), # One position per batch
            block=(num_threads,), # Number of threads per block as "advised" by the function above
            args=( # Pointers to all the tensors, we ensured they are contiguous.
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr()
            )
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # We don't handle the gradient for the feature indices and values, so
        # make sure it's not required.
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        # Retrieve the saved tensors.
        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        # Similar to the forward case
        kernel, num_threads = make_feature_transformer_slice_backward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            block=(num_threads,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr()
            )
        )

        # The order of returned values here is the same as the order of inputs to the forward pass.
        return None, None, weight_grad, bias_grad

class FeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Initialize in the same way nn.Linear would be initialized.
        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

    def forward(self, feature_indices, feature_values):
        # Use our FeatureTransformerSliceFunction for the forward pass.
        # Backward will automatically use the backward function from the FeatureTransformerSliceFunction class
        return FeatureTransformerSliceFunction.apply(feature_indices, feature_values, self.weight, self.bias)
```

## 架构和新方向

### 简单的HalfKP Stockfish架构

这是Stockfish使用的第一种架构。唯一的区别是，在这个文档中，我们使用的是HalfKP，它没有Shogi中多余的64个未使用特征。除此之外，没有什么新的东西，只是使用前面描述的基本组件来把事情放到正确的角度看。

![](img/HalfKP-40960-256x2-32-32-1.svg)

### HalfKAv2特征集

本文档中简要提到了HalfKA特征集，它是HalfKP的兄弟特征集。最初，它有一个小缺点，即浪费了一些空间。HalfKAv2是改进的版本，使用的空间少了8%，但其他方面完全相同。那么，这两者的区别是什么呢？让我们考虑一下给定我们的国王平方`S`的一组特征。通常在HalfKA中，有768个可能的特征，即`64*12`，因为有64个方格和12个棋子（类型+颜色）。但我们可以注意到，当我们的国王平方固定在`S`时，我们知道对手的国王不在`S` - 我们的国王只使用了给定的64个特征中的1个，而另一个国王只使用了给定的64个特征中的63个（减去我们的国王环，但这不重要），并且这两个集合是不相交的！因此，我们可以将两个棋子"合并为一个"，并将存储桶的数量从12减少到11，从而减少了大约8%的空间。然而，在应用因子化时必须小心，因为这种压缩需要被恢复，必须使用一个具有768个特征的完整`A`子集。否则可能会混淆国王位置，因为虽然这种压缩对单个`64*11`的桶是有效的，但在我们试图混合桶时，就不成立了，就像我们在特征化时所做的那样。

### HalfKAv2_hm特征集

这里的"hm"代表"水平镜像"。这个特征集基本上是HalfKAv2，但是棋盘被假定具有水平对称性。虽然这个假设在国际象棋中显然是不成立的，但在实践中却工作得很好。这个特征集的理念是，对于每个视角，将棋盘转换成我们的国王在e..h文件（出于约定，也可以是a..d文件）。知道只会使用一半的国王平方，使我们可以将输入特征的数量减半，从而有效地减半特征转换器的大小。自2021年8月初以来，Stockfish使用这个特征集来支持大的特征转换器，而无需不方便的大网络。这也被Scorpio使用，除了其他减小网络大小的方法。

让我们考虑一个例子，其中白色的国王在a1，黑色的国王在g1。对于白色的视角，需要对棋盘进行水平镜像，以将国王放在e..h文件内；对于黑色的视角，国王已经在e..h文件内。乍一看，这可能会在只为一个视角镜像棋盘的情况下造成视角的不一致，但实际上，这在实践中表现得非常好 - 力量差别几乎无法衡量。

### 特征转换器的一部分直接转发到输出。

通常，网络在学习高度物质失衡，甚至根本无法表示高度评估的情况时，会遇到困难。但我们可以帮助解决这个问题。我们已经为棋盘上的每一枚棋子积累了大约256个值，这是不是触动了你的思维？如果我们添加一个值，并指定它表示"PSQT"会怎么样呢？这就是我们要做的。我们只需要让特征转换器的权重行有257个值，并使用最后一个作为"PSQT"。在训练期间，我们可以通过初始化它为类似于好的PSQT值的东西来帮助它（但记得根据量化来调整它！）。但是我们有两个视角吗？那又怎么样？对，我们确实有，但是我们可以像这样平均它们：`(我们的 - 他们的) / 2`（记住他们必须被否定）。在训练器中处理它相当简单。

```python
wp = self.ft(w_in)
bp = self.ft(b_in)
w, wpsqt = torch.split(wp, wp.shape[1]-1, dim=1)
b, bpsqt = torch.split(bp, bp.shape[1]-1, dim=1)
[...]
y = self.output(l2_) + (wpsqt - bpsqt) * (us - 0.5)
```

我们也应该使用包含王特征的特征集，因为它提供了可能重要的额外PSQT值。所以我们将使用HalfKAv2。

![](img/HalfKAv2-45056-256x2P1x2-32-32-1.svg)

### 多个PSQT输出和多个子网络

到目前为止，所有网络都使用一个PSQT输出和一层堆栈（即Stockfish网络中的那部分-32-32-1；也就是特征变换器之后的部分）。但是，如果我们能用更多呢？我们需要找到一些易于计算的判别器来选择输出/层堆栈。一个这样的好判别器是棋子数量，因为它易于计算，游戏中表现相当稳定，而且棋子的数量可以显著改变我们对局面的看法。因此，让我们尝试基于`(棋子数量 - 1) / 4`的8个桶。

![](img/HalfKAv2-45056-256x2P8x2[-32-32-1]x8.svg)

但是在训练器中如何实现呢？"选择东西"对GPU并不友好，我们还要做批处理，对吧？的确如此，但是幸运的是，这些层非常小，所以我们可以评估所有的层，只选择结果！而且，多个`N`线性层可以被一个输出为`N`倍的单一线性层模拟。让我们看看它如何在PyTorch中实现：

```python
# 隐藏神经元的数量
L1 = 256
L2 = 32
L3 = 32

# 定义LayerStacks类
class LayerStacks(nn.Module):
    def __init__(self, count):
        super(LayerStacks, self).__init__()

        self.count = count
        # 层更大，对GPU非常有利
        self.l1 = nn.Linear(2 * L1, L2 * count)
        self.l2 = nn.Linear(L2, L3 * count)
        self.output = nn.Linear(L3, 1 * count)

        # 用于稍后缓存一些"魔法"操作
        self.idx_offset = None

        # 不要忘记根据你的喜好初始化层。
        # 每个层堆栈中的层都进行初始化可能是值得的，
        # 或者为层堆栈中的第一层引入一个因子。

    def forward(self, x, layer_stack_indices):
        # 预计算并缓存gather操作的偏移量
        if self.idx_offset == None or self.idx_offset.shape[0] != x.shape[0]:
            # 这就是"魔法"。对于批次中的每个位置，没有简单的方法从
            # 许多内容中收集只有一件事，但是我们可以将整个批次解释为
            # N * batch_size的输出，并修改layer_stack_indices指向
            # `N * i + layer_stack_indices`，其中`i`是位置索引。
            # 在这里我们预计算加法部分。这部分只包含值`N * i`
            self.idx_offset = torch.arange(0, x.shape[0] * self.count, self.count, device=layer_stack_indices.device)

        # 在这里我们将当前的索引添加到加法部分。
        indices = layer_stack_indices.flatten() + self.idx_offset

        # 评估整个层
        l1s_ = self.l1(x)
        # 将输出视为`N * batch_size`块
        # 根据我们之前计算的索引选择`batch_size`块。
        l1c_ = l1s_.view(-1, L2)[indices]
        # 我们本可以早点应用ClippedReLU，但这并不重要。
        l1y_ = torch.clamp(l1c_, 0.0, 1.0)

        # 对第二层做同样的操作。
        l2s_ = self.l2(l1y_)
        l2c_ = l2s_.view(-1, L3)[indices]
        l2y_ = torch.clamp(l2c_, 0.0, 1.0)

        # 对第三层也做同样的操作，但是没有夹紧操作，因为这是输出。
        l3s_ = self.output(l2y_)
        l3y_ = l3s_.view(-1, 1)[indices]

        return l3y_
```

处理PSQT输出会更容易，因为事实上，这是收集单个值的简单方法（我们之前不能使用它是因为我们正在收集整行）：

```python
wp = self.input(w_in)
bp = self.input(b_in)
w, wpsqt = torch.split(wp, wp.shape[1]-8, dim=1)
b, bpsqt = torch.split(bp, bp.shape[1]-8, dim=1)
[...]
psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
wpsqt = wpsqt.gather(1, psqt_indices_unsq)
bpsqt = bpsqt.gather(1, psqt_indices_unsq)
y = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)
```

## 历史上的Stockfish评估网络架构

### "SFNNv5" 架构

2022-05-14 - *

[Commit c079acc26f93acc2eda08c7218c60559854f52f0](https://github.com/official-stockfish/Stockfish/commit/c079acc26f93acc2eda08c7218c60559854f52f0)

![](img/SFNNv5_architecture_detailed_v2.svg)

### "SFNNv4" 架构

2022-02-10 - 2022-05-14

[Commit cb9c2594fcedc881ae8f8bfbfdf130cf89840e4c](https://github.com/official-stockfish/Stockfish/commit/cb9c2594fcedc881ae8f8bfbfdf130cf89840e4c)

![](img/SFNNv4_architecture_detailed_v2.svg)

### "SFNNv3" 架构

2021-08-15 - 2022-02-10

[Commit d61d38586ee35fd4d93445eb547e4af27cc86e6b](https://github.com/official-stockfish/Stockfish/commit/d61d38586ee35fd4d93445eb547e4af27cc86e6b)

![](img/SFNNv3_architecture_detailed_v2.svg)

### "SFNNv2" 架构

2021-05-18 - 2021-08-15

[Commit e8d64af1230fdac65bb0da246df3e7abe82e0838](https://github.com/official-stockfish/Stockfish/commit/e8d64af1230fdac65bb0da246df3e7abe82e0838)

![](img/SFNNv2_architecture_detailed_v2.svg)

### "SFNNv1" 架构

Also known as "Stockfish 12 architecture".

2020-08-06 - 2021-05-18

[Commit 84f3e867903f62480c33243dd0ecbffd342796fc](https://github.com/official-stockfish/Stockfish/commit/84f3e867903f62480c33243dd0ecbffd342796fc)

![](img/SFNNv1_architecture_detailed_v2.svg)

