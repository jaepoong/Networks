def vgg_params(model_name):
    """ VGG 파라미터 결정 """
    params_dict = {
        # 타입, 이미지 크기, 배치 정규화
        "vgg11": ("A", 224, False),
        "vgg13": ("B", 224, False),
        "vgg16": ("D", 224, False),
        "vgg19": ("E", 224, False),
        "vgg11_bn": ("A", 224, True),
        "vgg13_bn": ("B", 224, True),
        "vgg16_bn": ("D", 224, True),
        "vgg19_bn": ("E", 224, True),
    }
    return params_dict[model_name]




