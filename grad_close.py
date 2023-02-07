import os
import torch
from utils.args import *



class arm_calculator(object):
    def __init__(self, model, sample_num = 10):
        super(arm_calculator, self).__init__()
        self.sample_number = sample_num
        self.model = model

    def arm_calculate(self, video, tbar):
        arm_grad = []
        for i in range(self.sample_number):
            arm_grad.append(self.arm_grad_encoder(video, tbar))

        return sum(arm_grad)/self.sample_number

    def arm_grad_encoder(self, video, tbar):
        '''Given a video input, calculate the gradient of the VAE reconstruction
        loss w.r.t. the Bernoulli parameter using ARM. To make sure the implementation is correct,
        (1) Check tensor dimension
        (2)Test example: compare analytic form of gradient
        (polynomial, sin function and matrix multiplication) and estimated gradient.'''
        #Todo:(3)Compare with straght-through?
        # (4)How much complexity is reduced by using ARM?
        # Can that be implemented in a clean way? For example, input and function, return \nabla_input(function)
        batchsize, max_frames, feature_size = video.size()
        #MAKE SURE GRAD IS MULTIPLIED WITH T*CORRECTLY
        #tbar.size = [Batch, Hashsize]
        #Sample Unif[Batch, Hashsize]
        #Question: Should I sample U ONCE or BATCH TIMES? I think it doesnot really matters
        #tbar = model.encoder_forward(video)
        size = tbar.size()
        u = torch.rand(size).cuda()
        #tbar and u[Batch, Hashsize]
        #Calculate indicator[]
        Score2 = torch.sigmoid(tbar)
        Score1 = 1 - Score2
        Indicator1 = 2*(u>Score1).float() - 1
        Indicator2 = 2*(u<Score2).float() - 1
        #Decoding Indicator function
        frame1 = self.model.decoder_forward(Indicator1)
        frame2 = self.model.decoder_forward(Indicator2)
        #Size of frame[Batch, length, frame]
        mask_loss1 = torch.sum((frame1-video)**2, dim = (1, 2))/(batchsize*max_frames*feature_size)
        mask_loss2 = torch.sum((frame2-video)**2, dim = (1, 2))/(batchsize*max_frames*feature_size)
        #mask_loss[Batch], u [Batch, Hashsize]
        grad_tbar = ((mask_loss1 - mask_loss2).unsqueeze(1).repeat(1,size[1]))*(u - 1/2)
        #grad_tbar: [Batch, Hashsize]
        #Todo: Need to make sure grad_tbar is a constant here.
        #print('ARM Grad:', grad_tbar.detach())
        return grad_tbar.detach()

class u2g_calculator(object):
    def __init__(self, model, sample_num = 10):
        super(u2g_calculator, self).__init__()
        self.sample_number = sample_num
        self.model = model

    def u2g_calculate(self, video, tbar):
        u2g_grad = []
        for i in range(self.sample_number):
            u2g_grad.append(self.u2g_grad_encoder(video, tbar))

        return sum(u2g_grad)/self.sample_number

    def u2g_grad_encoder(self, video, tbar):
        '''Given a video input, calculate the gradient of the VAE reconstruction
        loss w.r.t. the Bernoulli parameter using ARM. To make sure the implementation is correct,
        (1) Check tensor dimension
        (2)Test example: compare analytic form of gradient
        (polynomial, sin function and matrix multiplication) and estimated gradient.'''
        #Todo:(3)Compare with straght-through?
        # (4)How much complexity is reduced by using ARM?
        # Can that be implemented in a clean way? For example, input and function, return \nabla_input(function)
        batchsize, max_frames, feature_size = video.size()
        #MAKE SURE GRAD IS MULTIPLIED WITH T*CORRECTLY
        #tbar.size = [Batch, Hashsize]
        #Sample Unif[Batch, Hashsize]
        #Question: Should I sample U ONCE or BATCH TIMES? I think it doesnot really matters
        #tbar = model.encoder_forward(video)
        size = tbar.size()
        u = torch.rand(size).cuda()
        #tbar and u[Batch, Hashsize]
        #Calculate indicator[]
        Score2 = torch.sigmoid(tbar)
        Score1 = 1 - Score2
        Indicator1 = (u>Score1).float()
        Indicator2 = (u<Score2).float()
        #Decoding Indicator function
        frame1 = self.model.decoder_forward(2*Indicator1 - 1)
        frame2 = self.model.decoder_forward(2*Indicator2 - 1)
        #Size of frame[Batch, length, frame]
        mask_loss1 = torch.sum((frame1-video)**2, dim = (1, 2))/(batchsize*max_frames*feature_size)
        mask_loss2 = torch.sum((frame2-video)**2, dim = (1, 2))/(batchsize*max_frames*feature_size)
        #mask_loss[Batch], u [Batch, Hashsize]
        #print('torch.sigmoid(torch.abs(tbar))', torch.sigmoid(torch.abs(tbar)).size())
        #print('(mask_loss1 - mask_loss2).unsqueeze(1).repeat(1,size[1])', (mask_loss1 - mask_loss2).unsqueeze(1).repeat(1,size[1]).size())
        #print('(Indicator1 - Indicator2)', (Indicator1 - Indicator2).size())
        grad_tbar = torch.sigmoid(torch.abs(tbar))*((mask_loss1 - mask_loss2).unsqueeze(1).repeat(1,size[1]))*(Indicator1 - Indicator2)/2
        #grad_tbar = ((mask_loss1 - mask_loss2).unsqueeze(1).repeat(1,size[1]))*(u - 1/2)
        #grad_tbar: [Batch, Hashsize]
        #Todo: Need to make sure grad_tbar is a constant here.
        #print('ARM Grad:', grad_tbar.detach())
        return grad_tbar.detach()

class close_form_calculator(object):
    def __init__(self, model, batch_size = 256, grad_batch_size = 1):
        super(close_form_calculator, self).__init__()
        self.batch_size = batch_size
        self.grad_batch_size = grad_batch_size
        self.weight_rc1 = model.sequence_restore.weight.data
        if model.sequence_restore.bias is not None:
            self.bias_rc1 = model.sequence_restore.bias.data
        else:
            self.bias_rc1 = torch.zeros(self.weight_rc1.size()[0]).cuda()

        self.weight_rc2 = model.restore.weight.data
        if model.restore.bias is not None:
            self.bias_rc2 = model.restore.bias.data
        else:
            self.bias_rc2 = torch.zeros(self.weight_rc2.size()[0]).cuda()


    def calculate_grad(self, video, tbar):
        anay_grad = []
        for i in range(self.batch_size//self.grad_batch_size+(self.batch_size%self.grad_batch_size)):
            if i+self.grad_batch_size<self.batch_size:
                anay_grad.append(self.anay_grad_encoder(video[i:i+self.grad_batch_size], torch.sigmoid(tbar[i:i+self.grad_batch_size])))
            else:
                anay_grad.append(self.anay_grad_encoder(video[i:self.batch_size], torch.sigmoid(tbar[i:self.batch_size])))

        anay_grad = torch.cat(anay_grad, dim = 0)
        return anay_grad

    def anay_grad_encoder(self, video, tbar):
        '''Given a video input, calculate the gradient of the VAE reconstruction
        loss w.r.t. the Bernoulli parameter using ARM. To make sure the implementation is correct,
        (1) Check tensor dimension
        (2)Test example: compare analytic form of gradient
        (polynomial, sin function and matrix multiplication) and estimated gradient.'''
        #Todo:(3)Compare with straght-through?
        # (4)How much complexity is reduced by using ARM?
        # Can that be implemented in a clean way? For example, input and function, return \nabla_input(function)
        #MAKE SURE GRAD IS MULTIPLIED WITH T*CORRECTLY
        #tbar.size = [Batch, Hashsize]
        Batch_size, Hashsize = tbar.size()
        _, length, Feature_size = video.size()
        #DIM0 = Batch*Hashsize
        DIM0 = Batch_size*Hashsize
        #tbar_expand.size = [Each Batch, Each Parameters, X, Y]
        tbar_unsqueeze = tbar.unsqueeze(2).unsqueeze(3)
        tbar_expand = tbar_unsqueeze.expand(-1, -1, Hashsize, Hashsize).transpose(1, 2)
        #Generate Selection matrix [Each Batch, Each Parameter, X, Y]
        #Generate one hot matrix [Hashsize, Hashsize]
        One_hot_matrix = torch.diag(torch.ones(Hashsize)).cuda()
        One_hot_matrix = One_hot_matrix.unsqueeze(0).unsqueeze(2).expand(Batch_size, -1, Hashsize, -1)
        #print('Onehot', One_hot_matrix)

        #Selected_matrix = [Each Batch, Each Parameter, X, Y]
        tbar_expand_view = tbar_expand.reshape(DIM0, Hashsize, Hashsize)
        One_hot_matrix_view = One_hot_matrix.reshape(DIM0, Hashsize, Hashsize)
        Selected_matrix = torch.mul(tbar_expand_view, One_hot_matrix_view).reshape(Batch_size, Hashsize, Hashsize, Hashsize)
        #Selected_matrix_trans = torch.permute(Selected_matrix, (0, 1, 3, 2))
        Selected_matrix_trans = Selected_matrix.transpose(2, 3)
        Sum_Selected_matrix = (Selected_matrix + Selected_matrix_trans)

        #Diagonal Selection matrix[Each Batch, Each parameter, X, Y]
        x_1 = torch.diag(torch.ones(Hashsize)).unsqueeze(2).cuda().float()
        x_2 = torch.diag(torch.ones(Hashsize)).unsqueeze(1).cuda().float()
        diag_select = torch.bmm(x_1, x_2).unsqueeze(0).expand(Batch_size, -1, -1, -1)

        Expect_bb = (Sum_Selected_matrix - Sum_Selected_matrix * diag_select + diag_select).reshape(DIM0, Hashsize, Hashsize).float()
        #print('bb_size', Expect_bb)
        #Expect_b
        Expect_b = torch.diag(torch.ones(Hashsize)).unsqueeze(1).cuda().float()
        #print('b_size',Expect_b)
        # weight_rc1 = model.sequence_restore.weight.data
        # bias_rc1 = model.sequence_restore.bias.data
        # weight_rc2 = model.restore.weight.data
        # bias_rc2 = model.restore.bias.data

        #print('weight_rc1:', model.sequence_restore.weight.data.size(), model.sequence_restore.weight.data)
        #print('bias_rc1:', model.sequence_restore.bias.data.size(), model.sequence_restore.bias.data)
        #print('weight_rc2:', model.restore.weight.data.size(), model.restore.weight.data)
        #print('bias_rc2:', model.restore.bias.data.size(), model.restore.bias.data)

        scaler = torch.mm(self.weight_rc1.transpose(0, 1), self.weight_rc1).sum()
        #print('scaler', scaler)

        bias_weight_term = torch.mm(torch.mm(self.weight_rc2, torch.ones((Hashsize, 1)).cuda()), self.bias_rc1.unsqueeze(0)) + torch.mm(self.bias_rc2.unsqueeze(1), torch.ones((1, length)).cuda())

        #print('FirstMatrix', torch.bmm(torch.bmm(weight_rc2.unsqueeze(0).expand(DIM0, -1, -1), Expect_bb), (weight_rc2.transpose(0, 1).unsqueeze(0).expand(DIM0, -1, -1))))

        First_term = scaler*torch.bmm(torch.bmm(self.weight_rc2.unsqueeze(0).expand(DIM0, -1, -1), Expect_bb), (self.weight_rc2.transpose(0, 1).unsqueeze(0).expand(DIM0, -1, -1))).diagonal(dim1=-1, dim2=-2).sum(-1).reshape(Batch_size, Hashsize)

        #print('First_term', First_term)
        #print('First_term_matrix:', torch.bmm(torch.bmm(self.weight_rc2.unsqueeze(0).expand(DIM0, -1, -1), Expect_bb), (self.weight_rc2.transpose(0, 1).unsqueeze(0).expand(DIM0, -1, -1))).diagonal(dim1=-1, dim2=-2))

        #print('2nd Matrix', 2*torch.bmm(torch.bmm((torch.mm(bias_weight_term, weight_rc1).unsqueeze(0).expand(Hashsize, -1, -1)), Expect_b), weight_rc2.transpose(0, 1).unsqueeze(0).expand(Hashsize, -1, -1)))

        Second_term = 2*torch.bmm(torch.bmm((torch.mm(bias_weight_term, self.weight_rc1).unsqueeze(0).expand(Hashsize, -1, -1)), Expect_b), self.weight_rc2.transpose(0, 1).unsqueeze(0).expand(Hashsize, -1, -1)).diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(0).expand(Batch_size, -1)

        #print('Second_term', Second_term)

        #print('3rdterm_debug:', video.transpose(1, 2).unsqueeze(1).expand(-1, Hashsize, -1, -1).reshape(DIM0, Feature_size, length).size())
        #print('3rdterm_debug2:', weight_rc1.unsqueeze(0).unsqueeze(1).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, length, 1).size())
        #print('3rdterm_debug3:', Expect_b.unsqueeze(0).expand(Batch_size, -1, -1, -1).reshape(DIM0, 1, Hashsize).size())
        #print('3rdterm_debug4:', weight_rc2.unsqueeze(0).unsqueeze(1).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, Hashsize, Feature_size).size())
        Third_term = 2*torch.bmm(torch.bmm(torch.bmm(video.transpose(1, 2).unsqueeze(1).expand(-1, Hashsize, -1, -1).reshape(DIM0, Feature_size, length), \
                                                     self.weight_rc1.unsqueeze(0).unsqueeze(0).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, length, 1)) \
                                           , Expect_b.unsqueeze(0).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, 1, Hashsize)), \
                                 self.weight_rc2.transpose(0, 1).unsqueeze(0).unsqueeze(0).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, Hashsize, Feature_size)) \
            .diagonal(dim1=-1, dim2=-2).sum(-1).reshape(Batch_size, Hashsize)

        #print('video', video)
        #print('video_trans', video.transpose(1, 2).unsqueeze(1).expand(-1, Hashsize, -1, -1).reshape(DIM0, Feature_size, length))
        #print('rc1_expand',weight_rc1.unsqueeze(0).unsqueeze(0).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, length, 1))
        #print('b_expand', Expect_b.unsqueeze(0).expand(Batch_size, -1, -1, -1).reshape(DIM0, 1, Hashsize))
        #print('rc2_expand', weight_rc2.transpose(0, 1).unsqueeze(0).unsqueeze(0).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, Hashsize, Feature_size))
        #print('3rd Matrix', 2*torch.bmm(torch.bmm(torch.bmm(video.transpose(1, 2).unsqueeze(1).expand(-1, Hashsize, -1, -1).reshape(DIM0, Feature_size, length),\
        #                                 weight_rc1.unsqueeze(0).unsqueeze(0).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, length, 1))\
        #                                 , Expect_b.unsqueeze(0).expand(Batch_size, -1, -1, -1).reshape(DIM0, 1, Hashsize)),\
        #                        weight_rc2.unsqueeze(0).unsqueeze(0).expand(Batch_size, Hashsize, -1, -1).reshape(DIM0, Hashsize, Feature_size)))

        #print('Third_term', -Third_term)
        #print('bias_weight_term', bias_weight_term.size())
        #print('First_term', First_term.size())
        #print('Second_term', Second_term.size())
        #print('Third_term', Third_term.size())



        #print('weight_rc2', model.)

        #print(model.sequence_restore(torch.ones(1).cuda()).unsqueeze(1))
        #print(model.sequence_restore(model.sequence_restore(torch.ones(1).cuda()).unsqueeze(1)).diagonal())
        #print(Sum_Selected_matrix[:, :, :, :])
        #torch.diagonal(Sum_Selected_matrix, dim1=2, dim2=3)
        #Result
        grad_tbar = (First_term + Second_term - Third_term)/(length*Feature_size)
        #Todo: Need to make sure grad_tbar is a constant here.
        #grad_tbar.size = [Batch, Hashsize]
        return grad_tbar.detach()
        #return grad_tbar.detach()


def arm_grad_sim_loss(video, tbar, bj):
    '''Given a video input, calculate the gradient of the similarity
    loss w.r.t. the Bernoulli parameter using ARM. To make sure the implementation is correct'''
    #Todo:(3)Compare with straght-through?
    # (4)How much complexity is reduced by using ARM?
    # Can that be implemented in a clean way? For example, input and function, return \nabla_input(function)
    batchsize, max_frames, feature_size = video.size()
    #MAKE SURE GRAD IS MULTIPLIED WITH T*CORRECTLY
    #tbar.size = [Batch, Hashsize]
    #Sample Unif[Batch, Hashsize]
    #Question: Should I sample U ONCE or BATCH TIMES? I think it doesnot really matters
    #tbar = model.encoder_forward(video)
    size = tbar.size()
    u = torch.rand(size).cuda()
    #tbar and u[Batch, Hashsize]
    #Calculate indicator[]
    Score2 = torch.sigmoid(tbar)
    Score1 = 1 - Score2
    Indicator1 = (u>Score1).float()
    Indicator2 = (u<Score2).float()
    #Decoding Indicator function

    #Indicator1.mul(bj)[Batch, Hashsize], sim[Batch]
    sim_1 = torch.sum(Indicator1.mul(bj), 1)/nbits
    sim_2 = torch.sum(Indicator2.mul(bj), 1)/nbits
    #print('size of sim:', sim_1.size())

    #Size of neighbor_loss [Batch]
    nei_loss1 = (  1*data["is_similar"].float()-sim_1)**2
    nei_loss2 = (  1*data["is_similar"].float()-sim_2)**2
    #print('size of neighbor loss:', nei_loss1.size())

    #mask_loss1 = torch.sum((frame1-video)**2, dim = (1, 2))/(max_frames*feature_size)
    #mask_loss2 = torch.sum((frame2-video)**2, dim = (1, 2))/(max_frames*feature_size)
    #mask_loss[Batch], u [Batch, Hashsize]
    grad_tbar = ((nei_loss1 - nei_loss2).unsqueeze(1).repeat(1,size[1]))*(u - 1/2)
    #grad_tbar: [Batch, Hashsize]
    #print('size of grad:', grad_tbar.size())
    #Todo: Need to make sure grad_tbar is a constant here.
    #print('ARM Grad:', grad_tbar.detach())
    return grad_tbar.detach()