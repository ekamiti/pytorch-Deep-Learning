# %%
import torch
# %%
class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.l1_in_dim = linear_1_in_features
        self.l1_out_dim = linear_1_out_features
        self.l2_in_dim = linear_2_in_features
        self.l2_out_dim = linear_2_out_features
        assert self.l1_out_dim == self.l2_in_dim

        self.f_function = f_function
        self.g_function = g_function

        def get_non_linear(func_name):
            if func_name == 'relu':
                return self.relu, self.grad_relu
            elif func_name == 'sigmoid':
                return lambda x: 1/(1 + torch.exp(-x)), self.grad_sigmoid
            elif func_name == 'identity':
                return lambda x: x, self.grad_identity
            else:
                raise NotImplementedError
        self.f, self.f_grad = get_non_linear(f_function)
        self.g, self.g_grad = get_non_linear(g_function)

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def relu(self, in_ : torch.Tensor):
        """
        Args:
            in_: tensor shape (batch_size, feature_len)
        Return:
            tensor shape (batch_size, feature_len)        
        """
        batch_size, feature_len = in_.size()
        ret = torch.max(in_, torch.zeros_like(in_))
        assert ret.size() == torch.Size([batch_size, feature_len])
        return ret

    def grad_relu(self, in_: torch.Tensor):
        """
        Args:
            in_: tensor shape (batch_size, feature_len)
        Return:
            tensor shape(batch_size, feature_len, feature_len)
        """
        batch_size, feature_len = in_.size()
        dxdz = [torch.diag((x_>0).float()) for x_ in in_]
        ret = torch.stack(dxdz)
        assert ret.size() == torch.Size([batch_size, feature_len, feature_len])
        return ret

    def grad_sigmoid(self, in_: torch.Tensor):
        """
        Args:
            in_: tensor shape (batch_size, feature_len)
        Return:
            tensor shape(batch_size, feature_len, feature_len)
        """
        batch_size, feature_len = in_.size()
        sig_prime = lambda e : torch.sigmoid(e) * (1 - torch.sigmoid(e))
        dxdz = [torch.diag(sig_prime(x_)) for x_ in in_]
        ret = torch.stack(dxdz)
        assert ret.size() == torch.Size([batch_size, feature_len, feature_len])
        return ret

    def grad_identity(self, in_):
        """
        Args:
            in_: tensor shape (batch_size, feature_len)
        Return:
            tensor shape(batch_size, feature_len, feature_len)
        """
        batch_size, feature_len = in_.size()
        dxdz = [torch.eye(x_.size()[0]) for x_ in in_]
        ret = torch.stack(dxdz)
        assert ret.size() == torch.Size([batch_size, feature_len, feature_len])
        return ret
    
    def grad_W(self, W, z):
        """
        Args:
            W: tensor shape(features_out, features_in)
            z: tensor shape(batch_size, features_in)
        Return:
            tensor shape(batch_size, features_out, features_out*features_in)
        """
        batch_size, features_in = z.size()
        features_out, W_features_in = W.size()
        assert W_features_in == features_in

        dim_y = W.size()[0]
        out = []
        for zi_val in z:
            dydw = []
            for y_idx in range(dim_y):
                dyi_dwi = None
                for wi_idx, _ in enumerate(W):
                    if wi_idx == y_idx:
                        if dyi_dwi is not None:
                            dyi_dwi = torch.cat((dyi_dwi, zi_val))
                        else:
                            dyi_dwi = zi_val
                    else:
                        if dyi_dwi is not None:
                            dyi_dwi = torch.cat((dyi_dwi, torch.zeros_like(zi_val)))
                        else:
                            dyi_dwi = torch.zeros_like(zi_val)
                dydw.append(dyi_dwi)
            out.append(torch.stack(dydw))
        ret = torch.stack(out)
        assert ret.size() ==\
            torch.Size([batch_size, features_out, features_out*features_in])
        return ret

    def grad_x(self, W):
        """
        Args:
            W: tensor shape(features_out, features_in)
        Return:
            tensor shape(features_out, features_in)
        """
        return W

    def grad_b(self, b):
        """
        Args:
            b: tensor shape(feature_len)
        Return:
            tensor shape(feature_len, feature_len)
        """
        b_dim = b.size()[0]
        ret = torch.eye(b_dim)
        assert ret.size() == torch.Size([b_dim, b_dim])
        return ret

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        Return:
            y_hat: tensor shape (batch_size, l2_out_features)
        """
        batch_size, input_dim = x.size()
        assert input_dim == self.l1_in_dim
        self.cache['x'] = x
        # z_1 shape (batch_size, linear_1_out_features)
        self.cache['z_1'] = x @ self.parameters['W1'].t() +\
            self.parameters['b1']
        assert self.cache['z_1'].size() ==\
            torch.Size([batch_size, self.l1_out_dim])
        # z_2 shape (batch_size, linear1_out_fea tures)
        self.cache['z_2'] = self.f(self.cache['z_1'])
        assert self.cache['z_2'].size() ==\
            torch.Size([batch_size, self.l2_in_dim])
        # z_3 
        self.cache['z_3'] = self.cache['z_2'] @ self.parameters['W2'].t() +\
            self.parameters['b2']
        assert self.cache['z_3'].size() ==\
            torch.Size([batch_size, self.l2_out_dim])
        self.cache['y_hat'] = self.g(self.cache['z_3'])
        assert self.cache['y_hat'].size() ==\
            torch.Size([batch_size, self.l2_out_dim])
        return self.cache['y_hat']

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape 
            (batch_size, linear_2_out_features)
        """
        batch_size, output_dim = dJdy_hat.size()
        assert output_dim == self.l2_out_dim
        # batch_size * l2_out * l2_out
        dyhat_dz3 = self.g_grad(self.cache['z_3'])
        assert dyhat_dz3.size() ==\
            torch.Size([batch_size, self.l2_out_dim, self.l2_out_dim])
        # batch_size x 1 x l2_out
        dJdy_hat = dJdy_hat.view(batch_size, 1, output_dim)
        # batch_size x 1 x l2_out 
        dJ_dz3 = dJdy_hat @ dyhat_dz3 
        assert dJ_dz3.size() == torch.Size([batch_size, 1, self.l2_out_dim])
        # l2_out x l2_in
        dz3_dz2 = self.grad_x(self.parameters['W2'])
        assert dz3_dz2.size() == torch.Size([self.l2_out_dim, self.l1_out_dim])
        # batch_size x 1 x l2_in  
        dJ_dz2 = dJ_dz3 @ dz3_dz2
        assert dJ_dz2.size() == torch.Size([batch_size, 1, self.l2_in_dim])
        # batch_size x l2_out x (l2_out * l2_in)
        dz3_dw2 = self.grad_W(self.parameters['W2'], self.cache['z_2'])
        assert dz3_dw2.size() ==\
            torch.Size([batch_size, self.l2_out_dim,
            self.l2_out_dim * self.l2_in_dim])
        # batch_size x 1 x (l2_out * l2_in)
        dJ_dw2 = dJ_dz3 @ dz3_dw2
        assert dJ_dw2.size() ==\
            torch.Size([batch_size, 1, self.l2_out_dim * self.l2_in_dim])
        # batch_size x l2_out x l2_in
        dJ_dw2 = dJ_dw2.view(batch_size, self.l2_out_dim, self.l2_in_dim)
        # l2_out x l2_out
        dz3_db2 = self.grad_b(self.parameters['b2'])
        assert dz3_db2.size() == torch.Size([self.l2_out_dim, self.l2_out_dim])
        # batch_size x 1 x l2_out
        dJ_db2 = dJ_dz3 @ dz3_db2
        assert dJ_db2.size() == torch.Size([batch_size, 1, self.l2_out_dim])
        dJ_db2 = dJ_db2.view(batch_size, self.l2_out_dim)
        # batch_size x 11_out x l1_out
        dz2_dz1 = self.f_grad(self.cache['z_1'])
        assert dz2_dz1.size() ==\
            torch.Size([batch_size, self.l1_out_dim, self.l1_out_dim])
        # batch_size x 1 x (l1_out == l2_in)
        dJ_dz1 = dJ_dz2 @ dz2_dz1
        assert dJ_dz1.size() ==\
            torch.Size([batch_size, 1, self.l1_out_dim])
        # batch_size x l1_out x (l1_out * l1_in)
        dz1_dw1 = self.grad_W(self.parameters['W1'], self.cache['x'])
        assert dz1_dw1.size() ==\
            torch.Size([batch_size, self.l1_out_dim,
                self.l1_out_dim * self.l1_in_dim])
        # batch_size x 1 x (l1_out * l1_in)
        dJ_dw1 = dJ_dz1 @ dz1_dw1
        assert dJ_dw1.size() ==\
            torch.Size([batch_size, 1, self.l1_out_dim * self.l1_in_dim])
        # batch_size x l1_out x l1_in
        dJ_dw1 = dJ_dw1.view(batch_size, self.l1_out_dim, self.l1_in_dim)
        # l1_out x l1_out
        dz1_db1 = self.grad_b(self.parameters['b1'])
        assert dz1_db1.size() ==\
            torch.Size([self.l1_out_dim, self.l1_out_dim])
        # batch_size x 1 x l1_out
        dJ_db1 = dJ_dz1 @ dz1_db1
        assert dJ_db1.size() == torch.Size([batch_size, 1, self.l1_out_dim])
        dJ_db1 = dJ_db1.view(batch_size, self.l1_out_dim)

        self.grads['dJdW1'] = torch.sum(dJ_dw1, 0)
        self.grads['dJdb1'] = torch.sum(dJ_db1, 0)
        self.grads['dJdW2'] = torch.sum(dJ_dw2, 0)
        self.grads['dJdb2'] = torch.sum(dJ_db2, 0)
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

# %%
def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    batch_size, vect_len = y.size()
    # ||y_hat-y||^2 = mean(sqrt(<y_hat-y,y_hat-y>)^2) = mean(<y_hat-y,y_hat-y>)
    diff = y_hat - y
    # sum over columns
    loss = torch.mean(torch.square(diff))
    #assert loss.size() == torch.Size([1])

    # dJdy_hat = 2 (y_hat-y)^T
    dJdy_hat = (2/(batch_size * vect_len)) * diff

    assert dJdy_hat.size() == torch.Size([batch_size, vect_len])
    return loss, dJdy_hat

# %%
def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # (1/K) * -(sum_1^K)([y_i*log(y_hat_i) + (1-y_i)*log(1-y_hat_i)])
    batch_size, vect_len = y.size()
    one_over_K = 1 / vect_len
    y_diff = 1 - y
    y_hat_diff = 1 - y_hat
    term1 = y * torch.log(y_hat)
    term2 = y_diff * torch.log(y_hat_diff)
    # sum over columns
    loss = -torch.mean(term1 + term2)

    ##assert loss.size() == torch.Size([1])

    # dydy_hat = (1/K*N) * (((1-y)/(1-y_hat)) - (y/y_hat))T dim -> 1xK
    term1 = y_diff / y_hat_diff
    term2 = y / y_hat
    dJdy_hat = (one_over_K / batch_size) * (term1 - term2)

    assert dJdy_hat.size() == torch.Size([batch_size, vect_len])
    return loss, dJdy_hat
# %%
