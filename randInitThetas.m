function thetas = randInitThetas(layer_dims)
% Randomly initialize the weight of a network with layers defined by layer_dims

thetas = [];

for l = 1:length(layer_dims)-1
	epsilon_init = sqrt(6) / sqrt(layer_dims(l) + layer_dims(l+1));
	thetas = [thetas; (rand(layer_dims(l+1), 1 + layer_dims(l)) * 2 * epsilon_init - epsilon_init)(:)];
end

end