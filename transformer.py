import torch

class attention(object):

    def __init__(self):
        self.linear_q = torch.nn.Linear()
        self.linear_k = torch.nn.Linear()
        self.linear_v = torch.nn.Linear()

    def forward(self, q, k, v):

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        att = torch.softmax(q * k)
        output = v * att

        return output


class transformer(object):

    def __init__(self):
        pass

    def forward(self, x):
        