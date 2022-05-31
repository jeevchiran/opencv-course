## Gaussian Blur / noise reduction
___
formula for gaussian distribution over 2 dimensional space
 - waited average of the pixels 
$$ G(x,y) = {1 \over 2\pi \sigma^2} e ^{ -(x^2+y^2) \over 2 \sigma^2} $$

## Bilateral Blur 
___
formula 
 - waited avaerage of neighboring pixels of similar intensity 
 - preserves edges 

 $$ BF[I_p] = { 1 \over W_p} \sum_ { q \in S } G_\sigma{_s} (||p-q||)G_\sigma{_r}{(I_p - I_q)}I_q $$

 $$ W_p = \sum_{ q \in S } G_\sigma{_s} (||p-q||)G_\sigma{_r}{(I_p - I_q)} $$