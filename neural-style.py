# -*- coding:utf-8 -*-
"""
Utlize finetuning learning on Neural Style. w.t. use trained model to do Neural Style.
The datasets putin ./datasets.
"""
import os
import argparse
import logging
import mxnet as mx
from mxnet.gluon.model_zoo import vision as models
# import mxnet/gulon/model_zoo/vision, utlize models. to load trained model.
# use models.function() to load trained model. 
# First, it will download the trained model, then load the pretrained model.
from mxnet import image # utlize mxnet/image module to read images and transform images.
import matplotlib.pyplot as plt # save images.
import net
import utils

# logging
log_file = "./neural_style.log"
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                   filename=log_file,
                   level=logging.INFO,
                   filemode='a+')
logging.getLogger().addHandler(logging.StreamHandler())

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="Utlize VGG19 on Neural Style.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_root", type=str, default="./datasets",
                        help='the root path of images.')
    parser.add_argument("--image_shape", type=tuple, default=(512, 512),
                        help='the shape of resized image.')
    parser.add_argument("--style_layers", type=list, default=[0,5,10,19,28],
                        help='Comma-separated list of layer names to use for style reconstruction.')
    parser.add_argument("--content_layers", type=list, default=[25],
                        help='Comma-separated list of layer names to use for content reconstruction.')
    parser.add_argument("--max_epochs", type=float, default=4000,
                        help='max epochs.')
    parser.add_argument("--lr", type=float, default=0.1,
                        help='lr.')
    parser.add_argument("--lr_decay_epoch", type=float, default=400,
                        help='lr decay epoch.')
    # The quality of generated image is related to the lr! like this set can obtain a good result.
    # lr=0.1.
    # some parameter's meaning can refer to https://github.com/jcjohnson/neural-style.
    parser.add_argument("--content_weights", type=list, default=[1],
                        help='weight the content reconstruction term.')
    parser.add_argument("--tv_weight", type=float, default=10.,
                        help='Weight of total-variation (TV) regularization.')
    
    
    args = parser.parse_args()

    # context
    ctx = utils.try_gpu()
    # ctx = mx.cpu()

    # neural style network
    pretrained_net = models.vgg19(pretrained=False)
    # Utlize class VGG + layers(11, 13, 16, 19. The number of conv layer) + 
    #     filters(num_kernels, e.g. [64, 128, 256, 512, 512]) to define a net. 
    pretrained_net.load_params(filename="/home/ly/mxnet/models/vgg19-f7134366.params", ctx=ctx)
    # Utlize get_model_file(name, root) to download the trained model. 
    # If you set pretrained=True, Every time you run the code, it will redownload the trained model.
    # So, in here set pretrained=False, load trained parameters directly!
    # content_layers + style_layers is: [25,0,5,10,19,28]. max(content_layers + style_layers) is 28.
    net = net.get_net(pretrained_net, args.content_layers, args.style_layers)
    # Use trained model to define our neural-style network. It contains 29 layer, e.g. conv + relu + bn....
    # Every layer of neural-style network has owned parameters, so don't initialize.
    net.collect_params().reset_ctx(ctx) 
    # parameters in ctx. This operate must do, ensure the parameters of network is in appointed ctx.
    # Otherwise, it maight have a error.
    # There not define trainer(the first argument is net.collect_params()), so we don't
    # train this net. We only train the content image.
    # Therefore, Neural Style's efficient is very high! Time is so short.

    # style image and content image.
    # There, we only load a pair of images. one is style image, the other is content image.
    content_img = image.imread(os.path.join(args.image_root, "tubingen.jpg"))
    style_img = image.imread(os.path.join(args.image_root, "the_scream.jpg"))
    # image.imread(): load a image, return NDArray. data format is BGR, w.t. HWC.
    # imshow
    '''
    plt.imshow(style_img.asnumpy()) # use asnumpy() transform to numpy ndarray.
    plt.show()
    plt.imshow(content_img.asnumpy())
    plt.show()
    '''
    # generate content image and features.
    content_x, content_y = utils.get_contents(net=net, ctx=ctx, content_img=content_img, image_shape=args.image_shape, 
        content_layers=args.content_layers, style_layers=args.style_layers)
    # content_x is the content image, content_y is the features of content image(list of features).
    # Our need is content_y. Features.
    # content_x is NDArray.

    # generate style image and features.
    style_x, style_y = utils.get_styles(net=net, ctx=ctx, style_img=style_img, image_shape=args.image_shape, 
        content_layers=args.content_layers, style_layers=args.style_layers)
    # style_x is the style image, style_y is the features of style image(list of features).
    # Our need is style_y. Features.
    # style_x is NDArray.

    x = content_x.copyto(ctx)
    # content_x is the content image
    # x is the content image.
    # In the process of training, it will learn x, w.t. the content image.
    # The content image to content + style image.
    x.attach_grad()
    # call x.attach_grad() to allocation ram space for gradient. w.t. 
    # x[:] -= args.lr * x.grad to learn content image to content+style image.

    y = utils.train(net=net, x=x, content_y=content_y, style_y=style_y, args=args)
    
    # save the result
    plt.imsave('result-lr{}.png'.format(str(args.lr)), utils.postprocess(y).asnumpy())
    # use asnumpy() transform NDArray to numpy array to save! Important!
    # use asnumpy()!