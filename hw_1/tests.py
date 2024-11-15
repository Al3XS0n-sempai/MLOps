def test_import():
    try:
        import relu_module
    except ModuleNotFoundError:
        pytest.fail("Module relu_module does not installed.")

    try:
        from relu_module import MyReLU
    except:
        pytest.fail("relu_module does not contain MyReLU.")

    from relu_module import MyReLU

    if not callable(MyReLU):
        pytest.fail("relu_module.MyReLU should be a function.")


def test_check_correct_type():
    from relu_module import MyReLU

    res: list[double] = MyReLU([1., -2., 3., 0., 5, -2., -1.])

    assert res == [1., 0., 3., 0., 5., 0., 0.]

    assert [] == MyReLU([])


def test_check_incorrect_types():
    from relu_module import MyReLU

    try:
        _ = MyReLU(1., 2., -1., 0.)

        pytest.fail("MyReLU should accept only list")
    except:
        pass

    try:
        _ = MyReLU(["1", "-1"])

        pytest.fail("MyReLU should accept only list")
    except:
        pass

    try:
        _ = MyReLU((1, -2, 3))

        pytest.fail("MyReLU should accept only list")
    except:
        pass

    try:
        _ = MyReLU({1, 2, 3, -1})

        pytest.fail("MyReLU should accept only list")
    except:
        pass
