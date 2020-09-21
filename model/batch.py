import torch

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
        self, 
        torch_batch,
        txt_pad_index,
        trg_pad_token,
        use_cuda        
        ):
        self.batch = torch_batch

        # Data information
        self.vid_names = self.batch.id
        
        # Text
        self.src_txt = self.batch.text[0]
        self.src_txt_mask = (self.src_txt != txt_pad_index).unsqueeze(1)
        self.src_length = self.batch.text[1]
        # self.num_src_tokens = self.src_length.sum().detach().clone().numpy()

        # Target pose sequences
        self.trg_seq = self.batch.landmark[0]
        self.trg_seq_mask = (self.trg_seq != trg_pad_token)[:, :, 0].unsqueeze(1)
        self.trg_length = self.batch.landmark[1]

        # Gloss
        self.gloss = self.batch.gloss[0]
        self.gloss_length = self.batch.gloss[1]
        self.num_gloss_tokens = self.gloss_length.sum().detach().clone().numpy()

        self.use_cuda = use_cuda

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        self.src_txt = self.src_txt.cuda()
        self.src_txt_mask = self.src_txt_mask.cuda()
        self.trg_seq = self.trg_seq.cuda()
        self.trg_seq_mask = self.trg_seq_mask.cuda()
        self.gloss = self.gloss.cuda()

        