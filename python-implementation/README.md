## Gaussian Blur / noise reduction
___
formula for gaussian distribution over 2 dimensional space
 - waited average of the pixels 
<!-- $$ 
G(x,y) = {1 \over 2\pi \sigma^2} e ^{ -(x^2+y^2) \over 2 \sigma^2} 
$$ --> 


<div align="center"><img style="background: white;" src="..\svg\09Qe272RJM.svg"></div>



## Bilateral Blur 
___
formula 
 - waited avaerage of neighboring pixels of similar intensity 
 - preserves edges 
 <!-- $$
 BF[I_p] = { 1 \over W_p} \sum_ { q \in S } G_\sigma{_s} (||p-q||)G_\sigma{  _r (I_p - I_q)}I_q
 $$ --> 

<div align="center"><img style="background: white;" src="..\svg\B3SyaOftf3.svg"></div>

 <!-- $$
 W_p = \sum_{ q \in S } G_\sigma{_s} (||p-q||)G_\sigma{  _r (I_p - I_q)}
 $$ --> 

<div align="center"><img style="background: white;" src="..\svg\8Nv4iH9hgh.svg"></div>

link : https://python.algorithmexamples.com/web/digital_image_processing/filters/bilateral_filter.html 

# Edge Detection

## Sobel
___

## canny 
link : https://www.geeksforgeeks.org/implement-canny-edge-detector-in-python-using-opencv/ 


# Morphology operations

