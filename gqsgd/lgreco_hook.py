import logging
import math

import numpy as np
import torch
import torch.distributed as dist

from . import default_hook as default
import gqsgd.allreduce

__all__ = [
    "PowerSGDState", "powerSGD_hook", "batched_powerSGD_hook"
]

logger = logging.getLogger(__name__)


def _orthogonalize(matrix, epsilon=0):
    """
    Decide between Gram-Schmidt or QR factorization to orthogonalize the matrix.
    QR factorization doesn't work with half-precision, but it is usually faster with a rank > 2.
    """
    assert len(matrix.shape) == 2 and matrix.shape[1] <= matrix.shape[0]

    rank = matrix.shape[1]
    dtype = matrix.dtype
    if rank <= 2 or dtype in [torch.float16, torch.bfloat16]:
        _orthogonalize_gram_schmidt(matrix, epsilon=epsilon)
    else:
        torch.linalg.qr(
            matrix,
            out=(
                matrix,
                torch.empty(rank, rank, device=matrix.device, dtype=dtype)
            )
        )


def _orthogonalize_gram_schmidt(matrix, epsilon=0):
    """
    Applies Gram-Schmidt procedure to orthogonalize a given 2D tensor.
    If epsilon is 0, this is equivalent to `torch.qr(matrix, out=(matrix, _))`,
    """
    num_cols = matrix.shape[1]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrix[:, i: i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input matrix covers the gradients of at least one entire layer in the neural network.
        if epsilon == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            try:
                col /= torch.norm(col)
            except ZeroDivisionError:
                logger.error(
                    "The matrix to be orthogonalized has at least a column of all 0s. Please set a small value such as 1e-8 "
                    "as `orthogonalization_epsilon` in PowerSGD state."
                )
                # Recover the values from NaNs to 0s.
                col.fill_(0.0)
        else:
            col /= torch.norm(col) + epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col


def _should_compress(
        num_rows, num_cols, matrix_approximation_rank, min_compression_rate
):
    """
    Returns a recommendation as to whether the 2D tensor described by the arguments is worth compressing,
    including statistics describing the expected savings from compression.  We consider a tensor worth
    compressing when ``min_compression_rate`` < uncompressed size / compressed size, where
    uncompressed size = ``num_rows`` * ``num_cols``,
    and compressed size = (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.
    The result of this function is a tuple of the form (compression_recommendation, uncompressed_el_count, compressed_el_count), where:
    compresion_recommendation is true if the tensor is worth compressing, and false otherwise (see above);
    uncompressed_el_count is the uncompressed element count, i.e. ``num_rows`` * ``num_cols``; and,
    compress_el_count is the element count after compression, i.e. (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.
    """  # noqa: B950
    uncompressed_size = num_rows * num_cols
    compressed_size = (num_rows + num_cols) * matrix_approximation_rank
    return (
        compressed_size * min_compression_rate < uncompressed_size,
        uncompressed_size,
        compressed_size,
    )


def _report_compression_stats(bucket, state):
    """
    Report compression stats at the frequency of `compression_stats_logging_frequency` specified in PowerSGD state.
    """
    if (
            bucket.is_last()
            and state.iter >= state.next_stats_report
    ):
        stats = state.compression_stats()
        logger.info(
            "Compression stats: iter {}, total before compression {}, total after compression {}, "
            "rate {}".format(state.iter, stats[1], stats[2], stats[0])
        )
        state.next_stats_report = state.iter + state.compression_stats_logging_frequency


class PowerSGDState(object):
    r"""
    Stores both the algorithm's hyperparameters and the internal state for all the gradients during the training.
    Particularly, ``matrix_approximation_rank`` and ``start_powerSGD_iter`` are the main hyperparameters that should be tuned by the user.
    For performance, we suggest to keep binary hyperparameters ``use_error_feedback`` and ``warm_start`` on.
    1. ``matrix_approximation_rank`` controls the size of compressed low-rank tensors, which determines the compression rate. The lower the rank, the stronger the compression.
        1.1. If ``matrix_approximation_rank`` is too low, the full model quality will need more training steps to reach or will never reach and yield loss in accuracy.
        1.2. The increase of ``matrix_approximation_rank`` can substantially increase the computation costs of the compression, and the accuracy may not be futher improved beyond a certain ``matrix_approximation_rank`` threshold.
    To tune ``matrix_approximation_rank``, we suggest to start from 1 and increase by factors of 2 (like an expoential grid search, 1, 2, 4, ...), until a satisfactory accuracy is reached. Typically only a small value 1-4 is used. For some NLP tasks (as shown in Appendix D of the original paper), this value has been increased to 32.
    2. ``start_powerSGD_iter`` defers PowerSGD compression until step ``start_powerSGD_iter``, and vanilla allreduce runs prior to step ``start_powerSGD_iter``. This hybrid scheme of **vanilla allreduce + PowerSGD** can effectively improve the accuracy, even a relatively small ``matrix_approximation_rank`` is used. This is because that, the beginning of training phase is usually very sensitive to inaccurate gradients, and compressing gradients too early may make the training quickly take a suboptimal trajectory, which can result in an irrecoverable impact on the accuracy.
    To tune ``start_powerSGD_iter``, we suggest to start with 10% of total training steps, and increase it until a satisfactory accuracy is reached. If there is a warm-up stage in the training, ``start_powerSGD_iter`` typically should be no less than the number of warm-up steps.
    3. ``min_compression_rate`` is the minimum compression rate required when a layer is compressed. Due to the computation overheads incurred by the compression, a tensor is worth compressing only if there can be sufficient saving in bandwidth, where ``(num_rows + num_cols) * matrix_approximation_rank * min_compression_rate < num_rows * num_cols``. If the specified compression rate threshold cannot be satisfied, the tensor will be directly allreduced without compression.
    Compression statistics are logged every ``compression_stats_logging_frequency`` iterations once PowerSGD compression starts.
    4. ``orthogonalization_epsilon`` can be a very small value (e.g., 1e-8) added to every normalized matrix column in orthogonalization step, to prevent div-by-zero error if any column has all 0s. If this can already be prevented (e.g., by batch normalization), an epsilon of 0 is recommended for accuracy.
    .. warning ::
        If error feedback or warm-up is enabled, the minimum value of ``start_powerSGD_iter`` allowed in DDP is 2.
        This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
        and this can conflict with any tensor memorized before the rebuild process.
    """  # noqa: B950

    __slots__ = [
        "process_group",
        # The fields below are the hyperparameters that often need to be tuned by the user.
        "matrix_approximation_rank",
        "start_powerSGD_iter",
        # The fields below are the hyperparameters that seldom need be tuned by the user.
        "min_compression_rate",
        "orthogonalization_epsilon",
        # The fields below are the binary hyperparameters recommended to be turned on for performance and accuracy.
        "use_error_feedback",
        "warm_start",
        # The fields below are internal state.
        "rng",
        "error_dict",
        "p_memory_dict",
        "q_memory_dict",
        "iter",
        # The fields below are for recording compression stats.
        "total_numel_before_compression",
        "total_numel_after_compression",
        "compression_stats_logging_frequency",
        "next_stats_report",
        # optimal compression parameters
        "opt_compress_param",
        "tensors_shape_n",
        "tensors_shape_m",
        "acc_grad",
        "adjust_freq",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "max_rank",
        "min_rank",
        "rank_range",
        "iter_rel",
        "tensors_to_adjust",
        "warmup_after_adjusting",
        "error_method",
        "error_method_iter",
        "acc_iter"
    ]

    def __init__(
            self,
            process_group,
            matrix_approximation_rank=1,
            start_powerSGD_iter=1_000,
            min_compression_rate=2,
            use_error_feedback=True,
            warm_start=True,
            orthogonalization_epsilon=0,
            random_seed=0,
            compression_stats_logging_frequency=10_000,
            adjust_freq=100,
            adjust_alpha=1,
            adjust_beta=2,
            adjust_gamma=0.5,
            adjust_delta=1e-2,
            warmup_after_adjusting=False,
            error_method='Power',
            error_method_iter=5,
    ):
        logger.info(
            "PowerSGD config: matrix_approximation_rank = {}; start_powerSGD_iter = {}; "
            "min_compression_rate = {}; orthogonalization_epsilon = {}; use_error_feedback = {}; warm_start = {}; "
            "random_seed = {}; compression_stats_logging_frequency = {}".format(
                matrix_approximation_rank,
                start_powerSGD_iter,
                min_compression_rate,
                orthogonalization_epsilon,
                use_error_feedback,
                warm_start,
                random_seed,
                compression_stats_logging_frequency,
            )
        )

        self.process_group = process_group
        self.matrix_approximation_rank = matrix_approximation_rank
        # Deferring PowerSGD compression util step 'start_powerSGD_iter' can have two advantages:
        # 1) It turns out that PowerSGD may lead to a non-trivial accuracy loss,
        # even if the matrix approximation rank is increased to a large value.
        # To mitigate the accuracy loss, a simple yet effective way is mixing vanilla allreduce
        # (or a more conservative compression such as FP16 compression) with PowerSGD.
        # 2) There is an internal optimization of rebuilding buckets process in DDP,
        # in order to save the memory space.
        # This step takes place after the first iteration.
        # However, this means that the shape of input bucketized tensors is subject to change,
        # which will complicate the implementations of error feedback and warm-up.
        # Running vanilla allreduce in the first few iterations can avoid this complexity.
        if (use_error_feedback or warm_start) and start_powerSGD_iter <= 1:
            raise ValueError(
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP."
            )
        self.start_powerSGD_iter = start_powerSGD_iter
        self.min_compression_rate = min_compression_rate
        # Error feedback is usually crucial for both for convergence and generalization,
        # because PowerSGD is a biased compressor,
        # i.e., compressing and decompressing a random gradient does not yield the original in expectation.
        # This mechanism requires a temporary copy of the input gradients,
        # so it increases the peak memory consumption by the size of the gradient tensor.
        # However, if the target matrices are known to be exactly low-ranked (instead of just low stable rank),
        # sometimes it is possible to converge to the optima without error feedback.
        # See: http://proceedings.mlr.press/v54/yurtsever17a/yurtsever17a.pdf
        self.use_error_feedback = use_error_feedback
        # Warm-start reuses P(s) and Q(s) from the previous iteration.
        # This can improve the approximation quality and hence improve the accuracy.
        # Additionally, by avoiding the initialization of these low-rank tensors at every step,
        # this can also accelerate training.
        # However, this is at the cost of extra memory.
        self.warm_start = warm_start
        # Can use a very small value to prevent div-by-zero error caused by orthogonalization of vanishing gradients.
        self.orthogonalization_epsilon = orthogonalization_epsilon
        # The purpose of this RNG is to generate different random seeds for initializing Q across iterations,
        # but in the same order for all the DDP replicas.
        # Different random seeds across iterations indicate different 'projections' of the gradients at different SGD steps.
        # If the same random projection is used,
        # there will be differences between the gradients that are never synchronized.
        self.rng = np.random.RandomState(random_seed)
        # Since there is only a single state instance for all the input buckets,
        # need to maintain a dictionary that maps each bucket index to the local error.
        self.error_dict = {}
        self.p_memory_dict = {}
        self.q_memory_dict = {}
        # Iteration/step in the training loop.
        self.iter = 0
        # Compression stats accumulators
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0
        # We'll report compression stats every 'compression_stats_logging_frequency' iterations
        # Note that we always report compression stats at least once.
        self.compression_stats_logging_frequency = max(
            1, compression_stats_logging_frequency
        )
        self.next_stats_report = 0
        # Adjuster Parameters Initialization
        self.opt_compress_param = []
        self.tensors_shape_n = []
        self.tensors_shape_m = []
        self.acc_grad = []
        self.tensors_to_adjust = []
        self.adjust_freq = adjust_freq
        self.alpha = adjust_alpha
        self.beta = adjust_beta
        self.gamma = adjust_gamma
        self.max_rank = int(np.ceil(self.matrix_approximation_rank * self.beta))
        self.min_rank = int(np.floor(self.matrix_approximation_rank * self.gamma))
        self.rank_range = self.max_rank - self.min_rank + 1
        self.delta = adjust_delta
        self.warmup_after_adjusting = warmup_after_adjusting
        self.error_method = error_method
        self.error_method_iter = error_method_iter
        self.acc_iter = 10

    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last():
            self.iter += 1
            self.iter_rel = self.iter - self.start_powerSGD_iter

        if self.iter == self.start_powerSGD_iter:
            logger.info(
                "Start to apply PowerSGD after {} iterations.".format(self.iter)
            )

    def compression_stats(self):
        r"""
        Returns the latest compression statistics as a tuple of the form (compress_rate, numel_before_compression, numel_after_compression), where:
        compress_rate is the effective compression rate i.e. (number of elements before compression) / (number of elements after compression);
        numel_before_compression is the total number of elements before compression was applied; and,
        numel_after_compression is the total number of elements after compression was applied.
        """  # noqa: B950
        compress_rate = (
            self.total_numel_before_compression / self.total_numel_after_compression
            if self.total_numel_after_compression > 0
            else 0
        )

        return (
            compress_rate,
            self.total_numel_before_compression,
            self.total_numel_after_compression,
        )


def powerSGD_hook(
        state: PowerSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    r"""
    This DDP communication hook implements PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:
    1. Views the input flattened 1D gradient tensor as a list of per-parameter tensors, and divides all the tensors into two groups:
        1.1 The tensors that should be compressed before allreduce, because the compression can give enough saving in bandwidth.
        1.2 Rest of the tensors will be directly allreduced without compression, including all the vector tensors (for biases).
    2. Handles uncompressed tensors:
        2.1. Allocate contiguous memory for those uncompressed tensors, and allreduces all the uncompressed tensors as a batch, without compression;
        2.2. Copies the individual uncompressed tensors from the contiguous memory back to the input tensor.
    3. Handles the tensors that should be compressed by PowerSGD compression:
        3.1. For each tensor M, creates two low-rank tensors P and Q for decomposing M,
        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;
        3.2. Computes each P in Ps, which is equal to MQ;
        3.3. Allreduces Ps as a batch;
        3.4. Orthogonalizes each P in Ps;
        3.5. Computes each Q in Qs, which is approximately equal to M^TP;
        3.6. Allreduces Qs as a batch;
        3.7. Computes each M among all the compressed tensors, which is approximately equal to PQ^T.
    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.
    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank``, ``start_powerSGD_iter``
            and ``min_compression_rate``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.
    Returns:
        Future handler of the communication, which updates the gradients in place.
    Example::
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1,
                                  start_powerSGD_iter=10, min_compression_rate=0.5)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """  # noqa: B950
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply PowerSGD after `start_powerSGD_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info(
                "A zero tensor of length {} that represents local error is created.".format(
                    total_length
                )
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=device, dtype=dtype
            )

        # Keep a copy of the input tensor,
        # so that we can compute the local error caused by compression later,
        # by comparing this copy and the input tensor updated after decompression.
        input_tensor_cp = torch.clone(input_tensor).detach()

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = bucket.gradients()

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    tensors_to_compress, uncompressed_tensors = [], []
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in tensors:
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        compress_test = _should_compress(
            n, m, matrix_approximation_rank, state.min_compression_rate
        )
        state.total_numel_before_compression += compress_test[1]
        if compress_test[0]:
            tensors_to_compress.append(matrix)
            total_Ps_size += n * matrix_approximation_rank
            total_Qs_size += m * matrix_approximation_rank
            state.total_numel_after_compression += compress_test[2]
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += compress_test[1]

    _report_compression_stats(bucket, state)

    # Step II: Handle uncompressed tensors.
    # Allocate contiguous memory for these tensors to allreduce efficiently.
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step III: Handle the tensors that should be compressed.
    # Allocate contiguous memory for Ps and Qs to allreduce efficiently.
    # If warm-start is enabled, reuse Ps and Qs from the previous iteration if possible.
    # The memory spaces of Ps and Qs need to be allocated in the first iteration when PowerSGD is applied.
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        # If warm-start is disabled, low-rank tensors will be initialized at every step.
        # Only log this if warm-start to avoid spamming.
        if state.warm_start:
            logger.info(
                "Allocating contiguous memory of length {} for Ps, and of length {} for Qs, respectively.".format(
                    total_Ps_size, total_Qs_size
                )
            )
        state.p_memory_dict[bucket_index] = torch.empty(
            total_Ps_size, device=device, dtype=dtype
        )
        state.q_memory_dict[bucket_index] = torch.empty(
            total_Qs_size, device=device, dtype=dtype
        )

    # DP adjuster
    # Accumulate Gradients
    if dist.get_rank() == 0:
        if state.iter_rel % state.adjust_freq == 0 and bucket_index == 0:
            state.tensors_to_adjust = []
            bucket_idx = 0
            for bucket_state in state.acc_grad:
                for layer_idx in range(len(bucket_state)):
                    state.tensors_to_adjust.append(state.acc_grad[bucket_idx][layer_idx])
                bucket_idx += 1
            state.acc_grad = []

        if state.iter_rel % state.adjust_freq >= state.adjust_freq - state.acc_iter:

            if len(state.acc_grad) <= bucket_index:
                state.acc_grad.append([])

            i = 0
            for tensor in tensors_to_compress:
                tensor_n = torch.mul(tensor, state.delta / state.acc_iter)
                if len(state.acc_grad[bucket_index]) > i:
                    torch.add(state.acc_grad[bucket_index][i], tensor_n, out=state.acc_grad[bucket_index][i])
                else:
                    state.acc_grad[bucket_index].append(tensor_n)
                i += 1

    # Initialize opt_compress_param and tensors_shapes
    if state.iter_rel == 0:
        opt_compress_param = []
        tensors_shape_n = []
        tensors_shape_m = []

        for tensor in tensors_to_compress:
            opt_compress_param.append(state.matrix_approximation_rank)
            n, m = tensor.shape
            tensors_shape_n.append(n)
            tensors_shape_m.append(m)

        state.opt_compress_param.append(opt_compress_param)
        state.tensors_shape_n.append(tensors_shape_n)
        state.tensors_shape_m.append(tensors_shape_m)
        
        if dist.get_rank() == 0:
            #print("Ns: ", tensors_shape_n)
            #print("Ms: ", tensors_shape_m)
            logger.info("Ns: {}".format(tensors_shape_n))
            logger.info("Ms: {}".format(tensors_shape_m))

    # DP Algorithm
    DPBUCKETS = 10000

    if state.iter_rel % state.adjust_freq == 0 and dist.get_rank() == 0 and bucket_index == 0 and state.iter_rel > 0:
        tensors_to_adjust = state.tensors_to_adjust
        # Build the Error Table using torch.svdvals()
        rank_range = state.rank_range
        min_rank = state.min_rank
        max_rank = state.max_rank

        num_layers = len(tensors_to_adjust)
        compression_errors = np.zeros([num_layers, rank_range])
        compressed_sizes = np.zeros([num_layers, rank_range])

        i = 0
        for tensor in tensors_to_adjust:
            n, m = tensor.shape
            r = min(n, m, max_rank)
            compressed_sizes[i, :r - min_rank + 1] = np.arange(min_rank, r + 1) * (n + m)
            # why "+1"? so that the algorithm returns the true rank!
            compressed_sizes[i, r - min_rank + 1:] = r * (n + m) + 1

            # Vectorized Implementation of SVD
            if state.error_method == 'SVD':
                s = torch.linalg.svdvals(tensor)
                ss = torch.square(s)
                ut = torch.triu(torch.full((r, min(n, m)), 1.0, device='cuda'), diagonal=1)
                compression_errors[i, :] = torch.sqrt(torch.matmul(ut, torch.t(ss))).cpu()[min_rank - 1:r]

            if state.error_method == 'Power':
                for r in range(min_rank, min(n, m, max_rank) + 1):
                    q = torch.randn(m, r, device='cuda')
                    p = torch.empty(n, r, device='cuda')
                    for k in range(state.error_method_iter):
                        _orthogonalize(q)
                        torch.matmul(tensor, q, out=p)
                        _orthogonalize(p)
                        torch.matmul(tensor.t(), p, out=q)
                    t = torch.empty(n, m, device='cuda')
                    torch.matmul(p, q.t(), out=t)
                    compression_errors[i, r - min_rank] = torch.dist(t, tensor)
            i += 1

        ## Discretize the Error Table
        static_error = np.sum(compression_errors[:, state.matrix_approximation_rank - state.min_rank])
        target_score = state.alpha * static_error

        ## Fixing the NaN Problem;
        compression_errors = np.minimum(compression_errors, target_score + 1)
        np.nan_to_num(compression_errors, nan=target_score + 1)

        bucket_size = target_score / DPBUCKETS
        compression_errors = np.minimum(np.maximum(np.ceil(compression_errors / bucket_size), 1), DPBUCKETS)

        np.nan_to_num(compression_errors, nan=DPBUCKETS + 1)
        compression_errors = compression_errors.astype(int)

        # Fill the DP table
        num_buckets = DPBUCKETS + 1
        num_values = state.rank_range

        DP = np.full((num_layers, DPBUCKETS + 1), float('inf'))
        PD = np.full((num_layers, DPBUCKETS + 1), -1)

        # Initialize the Table for the First Layer
        for bucket_idx in range(num_buckets):
            for val_idx in range(num_values):
                if bucket_idx > compression_errors[0, val_idx]:
                    tmp = compressed_sizes[0, val_idx]
                    if DP[0, bucket_idx] > tmp:
                        PD[0, bucket_idx] = val_idx
                        DP[0, bucket_idx] = tmp

        # Fill the Rest Recursively
        # Vectorized Implementation
        for layer_idx in range(1, len(DP)):
            for val_idx in range(num_values):
                comp_size = compressed_sizes[layer_idx][val_idx]
                comp_error = compression_errors[layer_idx][val_idx]
                tmp = DP[layer_idx - 1][:-comp_error] + comp_size
                better = tmp < DP[layer_idx][comp_error:]
                if np.sum(better):
                    DP[layer_idx][comp_error:][better] = tmp[better]
                    PD[layer_idx][comp_error:][better] = val_idx

        # Finding the Optimal Compression Scheme
        opt_compression_error = np.argmin(DP[-1, :])
        total_error = opt_compression_error

        if total_error <= DPBUCKETS:
            opt_compress_param = []
            # Build the Solution from the Table
            for layer_idx in range(len(DP) - 1, -1, -1):
                opt_compress_param.append(PD[layer_idx][total_error] + state.min_rank)
                total_error -= compression_errors[layer_idx, PD[layer_idx][total_error]]
            opt_compress_param.reverse()

            # Save the Optimal Compression Scheme into the State
            i = 0
            j = 0
            for bucket_state in state.opt_compress_param:
                for layer_idx in range(len(bucket_state)):
                    state.opt_compress_param[i][layer_idx] = opt_compress_param[j]
                    j += 1
                i += 1

        #print("optimal compression parameters = ", state.opt_compress_param)
        logger.info("optimal compression parameters = {}".format(state.opt_compress_param))

    if state.iter_rel % state.adjust_freq == 0 and bucket_index == 0 and state.iter_rel > 0:
        # Broadcast the Optimal Compression to All Workers
        dist.broadcast_object_list(state.opt_compress_param, src=0)

        # Recalculate Size of Ps and Qs
        bucket_idx = 0
        for bucket_state in state.opt_compress_param:
            total_Ps_size = 0
            total_Qs_size = 0
            for layer_idx in range(len(bucket_state)):
                n = state.tensors_shape_n[bucket_idx][layer_idx]
                m = state.tensors_shape_m[bucket_idx][layer_idx]
                matrix_approximation_rank = min(n, m, state.opt_compress_param[bucket_idx][layer_idx])
                total_Ps_size += n * matrix_approximation_rank
                total_Qs_size += m * matrix_approximation_rank

            # Reinitialize Ps and Qs with Correct Shapes!
            state.p_memory_dict[bucket_idx] = torch.empty(
                total_Ps_size, device=device, dtype=dtype
            )

            state.q_memory_dict[bucket_idx] = torch.randn(
                total_Qs_size, device=device, dtype=dtype
            )

            bucket_idx += 1

            # applying few warmup power step for all workers
            if state.warmup_after_adjusting:
                ps = []
                qs = []
                p_idx = 0
                q_idx = 0
                for layer_idx in range(len(tensors_to_compress)):
                    n = state.tensors_shape_n[bucket_index][layer_idx]
                    m = state.tensors_shape_m[bucket_index][layer_idx]
                    matrix_approximation_rank = min(n, m, state.opt_compress_param[bucket_index][layer_idx])
                    ps.append(
                        state.p_memory_dict[bucket_index][
                        p_idx: p_idx + n * matrix_approximation_rank
                        ].view(n, matrix_approximation_rank)
                    )
                    qs.append(
                        state.q_memory_dict[bucket_index][
                        q_idx: q_idx + m * matrix_approximation_rank
                        ].view(m, matrix_approximation_rank)
                    )
                    p_idx += n * matrix_approximation_rank
                    q_idx += m * matrix_approximation_rank
                    layer_idx += 1

                for tensor, q, p in zip(tensors_to_compress, qs, ps):
                    for k in range(state.error_method_iter):
                        _orthogonalize(q)
                        torch.matmul(tensor, q, out=p)
                        _orthogonalize(p)
                        torch.matmul(tensor.t(), p, out=q)

    # Create Ps and Qs that point to the allocated memory.
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0

    # Read the Ranks from the state
    i = 0
    for tensor in tensors_to_compress:
        n, m = tensor.shape
        matrix_approximation_rank = state.opt_compress_param[bucket_index][i]
        matrix_approximation_rank = min(n, m, matrix_approximation_rank)
        i += 1

        ps.append(
            state.p_memory_dict[bucket_index][
            p_idx: p_idx + n * matrix_approximation_rank
            ].view(n, matrix_approximation_rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
            q_idx: q_idx + m * matrix_approximation_rank
            ].view(m, matrix_approximation_rank)
        )
        p_idx += n * matrix_approximation_rank
        q_idx += m * matrix_approximation_rank

    # If warm-start is enabled, reuse Qs from the previous iteration if possible and skip filling random values.
    # The exception is the first iteration when PowerSGD is applied.
    if not need_randomize_qs:
        for q in qs:
            _orthogonalize(q, state.orthogonalization_epsilon)
    else:
        with torch.random.fork_rng(devices=[]):
            # Fork this RNG to avoid changing the seed globally and affecting the random sampling anywhere else in the training.
            # The seed makes sure that the initial random values are the same across all the DDP replicas.
            # This seed should differ at every step.
            # Since it is very slow to fork RNG state across all the CUDA devices,
            # only fork on CPU and then move the generated tensor to the CUDA device (by overwriting q).
            torch.manual_seed(state.rng.randint(1_000_000_000))
            for q in qs:
                q.copy_(
                    torch.randn(
                        *q.shape,
                        device="cpu",
                        dtype=dtype,
                    )
                )
                _orthogonalize(q, state.orthogonalization_epsilon)

    # Compute Ps.
    for tensor, q, p in zip(tensors_to_compress, qs, ps):
        torch.matmul(tensor, q, out=p)
# Old Code
    # allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
    #     uncompressed_tensors_memory, group=group_to_use, async_op=True
    # ).get_future()

    # def unpack_uncompressed_tensors_and_allreduce_ps(fut):
    #     uncompressed_tensors_memory = fut.value()[0].div_(world_size)
    #     idx = 0
    #     for tensor in uncompressed_tensors:
    #         tensor.copy_(
    #             uncompressed_tensors_memory[idx: idx + tensor.numel()].view_as(tensor)
    #         )
    #         idx += tensor.numel()

    #     # Since these Ps will be orthogonalized later, no need to divide them by world size.
    #     return (
    #         dist.all_reduce(
    #             state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
    #         )
    #             .get_future()
    #             .wait()[0]
    #     )

    # def compute_qs(fut):
    #     state.p_memory_dict[bucket_index] = fut.value()
    #     for p in ps:
    #         _orthogonalize(p, state.orthogonalization_epsilon)

    #     # Compute Qs.
    #     for tensor, p, q in zip(tensors_to_compress, ps, qs):
    #         torch.matmul(tensor.t(), p, out=q)

    #     # TODO: The above procedure does two matmul+allreduce steps per iteration --
    #     # one left multiplication and one right multiplication.
    #     # For warm-start, can take one such step at a time, and alternate between them.

    #     # Allreduce Qs.
    #     return (
    #         dist.all_reduce(
    #             state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
    #         )
    #             .get_future()
    #             .wait()[0]
    #     )

    # def decompress(fut):
    #     state.q_memory_dict[bucket_index] = fut.value().div_(world_size)

    #     for p, q, tensor in zip(ps, qs, tensors_to_compress):
    #         torch.matmul(p, q.t(), out=tensor)
    #     if torch.cuda.is_available():
    #         torch.cuda.synchronize(device)

    #     if state.use_error_feedback:
    #         # Memorize the local errors.
    #         state.error_dict[bucket_index] = input_tensor_cp - input_tensor
    #     if not state.warm_start:
    #         state.p_memory_dict.clear()
    #         state.q_memory_dict.clear()

    #     state.maybe_increase_iter(bucket)

    #     return input_tensor

    # return (
    #     allreduce_contiguous_uncompressed_tensors_fut.then(
    #         unpack_uncompressed_tensors_and_allreduce_ps
    #     )
    #         .then(compute_qs)
    #         .then(decompress)
    # )

# Tree Allreduce
    # This allreduce is only applied to uncompressed tensors,
    # so it should have been kicked off before the above computation on the compressed tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    gqsgd.allreduce.tree_allreduce(uncompressed_tensors_memory,exponential = False)
    fut = torch.futures.Future()
    fut.set_result(uncompressed_tensors_memory)
    def unpack_uncompressed_tensors_and_allreduce_ps(fut):
        uncompressed_tensors_memory = fut.value().div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                uncompressed_tensors_memory[idx: idx + tensor.numel()].view_as(tensor)
            )
            idx += tensor.numel()

        # Since these Ps will be orthogonalized later, no need to divide them by world size.
        gqsgd.allreduce.tree_allreduce(state.p_memory_dict[bucket_index],exponential = False)
        fut = torch.futures.Future()
        fut.set_result(state.p_memory_dict[bucket_index])
        return fut

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()
        for p in ps:
            _orthogonalize(p, state.orthogonalization_epsilon)

        # Compute Qs.
        for tensor, p, q in zip(tensors_to_compress, ps, qs):
            torch.matmul(tensor.t(), p, out=q)

        # TODO: The above procedure does two matmul+allreduce steps per iteration --
        # one left multiplication and one right multiplication.
        # For warm-start, can take one such step at a time, and alternate between them.

        # Allreduce Qs.
        gqsgd.allreduce.tree_allreduce(state.q_memory_dict[bucket_index], exponential = False)
        fut = torch.futures.Future()
        fut.set_result(state.q_memory_dict[bucket_index])
        return fut

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)

        for p, q, tensor in zip(ps, qs, tensors_to_compress):
            torch.matmul(p, q.t(), out=tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        if state.use_error_feedback:
            # Memorize the local errors.
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()

        state.maybe_increase_iter(bucket)

        return input_tensor
    fut = unpack_uncompressed_tensors_and_allreduce_ps(fut)
    fut = compute_qs(fut)
    result = decompress(fut)
    fut = torch.futures.Future()
    fut.set_result(result)
    return fut