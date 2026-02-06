"""Test that all optimization modules can be imported successfully."""


class TestOptimizationImports:
    """Test imports for optimization package modules."""

    def test_gemm_converter_imports(self):
        """Test GEMM converter module imports."""
        from onnx4deeploy.optimization.gemm_converter import (
            add_c_to_gemm,
            convert_fusedmatmul_to_no_bias_gemm,
            convert_gemm_to_transpose_matmul,
            fuse_matmul_add_to_gemm,
            unify_gemm_input_dims,
        )

        assert callable(add_c_to_gemm)
        assert callable(convert_gemm_to_transpose_matmul)
        assert callable(unify_gemm_input_dims)
        assert callable(fuse_matmul_add_to_gemm)
        assert callable(convert_fusedmatmul_to_no_bias_gemm)

    def test_gradient_optimizer_imports(self):
        """Test gradient optimizer module imports."""
        from onnx4deeploy.optimization.gradient_optimizer import (
            process_layernormgrad_nodes,
            remove_softmax_grad_loss_inputs,
            rename_softmaxgrad_op,
            run_optmization_remove_biasgelugrad,
        )

        assert callable(run_optmization_remove_biasgelugrad)
        assert callable(rename_softmaxgrad_op)
        assert callable(remove_softmax_grad_loss_inputs)
        assert callable(process_layernormgrad_nodes)

    def test_graph_cleaner_imports(self):
        """Test graph cleaner module imports."""
        from onnx4deeploy.optimization.graph_cleaner import (
            fix_layernorm_output,
            modify_conflict_outputs,
            optimize_softmax_axis,
            remove_identity_nodes,
            remove_softmax_loss_outputs,
            replace_biasgelu_with_gelu_add,
            run_optmization_remove_biasgelu,
        )

        assert callable(remove_identity_nodes)
        assert callable(remove_softmax_loss_outputs)
        assert callable(modify_conflict_outputs)
        assert callable(fix_layernorm_output)
        assert callable(replace_biasgelu_with_gelu_add)
        assert callable(run_optmization_remove_biasgelu)
        assert callable(optimize_softmax_axis)

    def test_shape_optimizer_imports(self):
        """Test shape optimizer module imports."""
        from onnx4deeploy.optimization.shape_optimizer import (
            convert_reducesum_axes_to_attr,
            convert_reducesum_to_reshape,
            convert_squeeze_unsqueeze_input_to_attr,
            convert_sum_to_add,
            optimize_reshape_fusion,
        )

        assert callable(optimize_reshape_fusion)
        assert callable(convert_reducesum_to_reshape)
        assert callable(convert_squeeze_unsqueeze_input_to_attr)
        assert callable(convert_reducesum_axes_to_attr)
        assert callable(convert_sum_to_add)

    def test_node_utils_imports(self):
        """Test node utils module imports."""
        from onnx4deeploy.optimization.node_utils import (
            add_backward_markers_to_nodes,
            annotate_node_names_with_type,
            convert_layernorm_to_groupnorm,
            process_onnx_model_name_with_type,
            split_gn_to_single_stat_array,
        )

        assert callable(annotate_node_names_with_type)
        assert callable(process_onnx_model_name_with_type)
        assert callable(add_backward_markers_to_nodes)
        assert callable(convert_layernorm_to_groupnorm)
        assert callable(split_gn_to_single_stat_array)

    def test_model_optimizer_imports(self):
        """Test model optimizer module imports."""
        from onnx4deeploy.optimization.model_optimizer import (
            run_onnx_optimization,
            run_onnx_optimization_infer,
        )

        assert callable(run_onnx_optimization_infer)
        assert callable(run_onnx_optimization)

    def test_train_optimizer_imports(self):
        """Test training optimizer module imports."""
        from onnx4deeploy.optimization.train_optimizer import run_train_onnx_optimization

        assert callable(run_train_onnx_optimization)

    def test_optimization_package_imports(self):
        """Test importing from optimization package __init__."""
        from onnx4deeploy.optimization import (
            add_c_to_gemm,
            optimize_reshape_fusion,
            remove_identity_nodes,
            run_onnx_optimization,
            run_train_onnx_optimization,
        )

        assert callable(add_c_to_gemm)
        assert callable(remove_identity_nodes)
        assert callable(optimize_reshape_fusion)
        assert callable(run_onnx_optimization)
        assert callable(run_train_onnx_optimization)


class TestUtilsImports:
    """Test imports for utils package modules."""

    def test_node_naming_imports(self):
        """Test node naming module imports."""
        from onnx4deeploy.utils.node_naming import (
            make_c_name,
            rename_and_save_onnx,
            rename_nodes,
            rename_onnx_nodes,
        )

        assert callable(make_c_name)
        assert callable(rename_onnx_nodes)
        assert callable(rename_and_save_onnx)
        assert callable(rename_nodes)


class TestIOImports:
    """Test imports for I/O package modules."""

    def test_config_loader_imports(self):
        """Test config loader module imports."""
        from onnx4deeploy.io.config_loader import load_config, load_train_config

        assert callable(load_config)
        assert callable(load_train_config)

    def test_model_io_imports(self):
        """Test model I/O module imports."""
        from onnx4deeploy.io.model_io import compare_onnx_models

        assert callable(compare_onnx_models)


class TestTransformImports:
    """Test imports for transform package modules."""

    def test_model_transform_imports(self):
        """Test model transform module imports."""
        from onnx4deeploy.transform.model_transform import (
            ensure_all_tensor_shapes,
            fix_shared_initializers_by_node_name,
            randomize_layernorm_params,
            randomize_onnx_initializers,
            split_convgrad_nodes,
            type_inference,
        )

        assert callable(split_convgrad_nodes)
        assert callable(randomize_layernorm_params)
        assert callable(fix_shared_initializers_by_node_name)
        assert callable(randomize_onnx_initializers)
        assert callable(type_inference)
        assert callable(ensure_all_tensor_shapes)


class TestTestingImports:
    """Test imports for testing package modules."""

    def test_pytorch_utils_imports(self):
        """Test PyTorch utils module imports."""
        from onnx4deeploy.testing.pytorch_utils import (
            create_test_input_output_pytorch,
            debug_grad_hook,
            debug_hook,
        )

        assert callable(create_test_input_output_pytorch)
        assert callable(debug_hook)
        assert callable(debug_grad_hook)
