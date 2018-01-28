# Neural Style
* This is a gulon implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.
* Borrowed code from https://zh.gluon.ai/chapter_computer-vision/neural-style.html.

## Install Required Packages
First ensure that you have installed the following required packages:
* mxnet0.12.0 ([instructions](http://mxnet.incubator.apache.org/install/index.html)). Maybe other version is ok. When you install mxnet, the gulon API will also install.
* Opencv ([instructions](https://github.com/opencv/opencv)). Here is opencv-2.4.9.

## More Qualitative results
The paper presents an algorithm for combining the content of one image with the style of another image using
convolutional neural networks. Here is some neural style results.

<div align="center">
 <img src="https://drive.google.com/file/d/11TXR-ZtjQB_Z6ozUFyUlrhKnSzEhs5kk/view?usp=sharing" height="223px">
 <img src="https://drive.google.com/open?id=1jzTgWuuFXsvbglkfIPfMrJ1IeoaQvb62" height="223px">
 <img src="results/hoovertowernight_starry_night_lr-0.1.png" width="710px">
</div>
