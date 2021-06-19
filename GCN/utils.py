import torch


def torch_delete(tensor, indices):
    if tensor.numel() == 0:
        return tensor
    mask = torch.ones(tensor.size(), dtype=torch.bool)
    mask[indices] = False
    return torch.masked_select(tensor, mask).reshape(tensor.size(0) - len(indices), -1)


def edge_type_2_reaction_vector(edge_type):
    return [edge_type.like, edge_type.reply, edge_type.retweet, edge_type.retweet_comment]


def reaction_list_2_int(r_list):
    return int("".join([str(int(x)) for x in r_list]), 2)


def reactions(proto_edge_types):
    reaction_vector = []
    reaction_int = []
    for i in range(len(proto_edge_types)):
        proto_edge_type = proto_edge_types[i]

        reaction_v = edge_type_2_reaction_vector(proto_edge_type)
        reaction_i = reaction_list_2_int(reaction_v)

        reaction_vector.append(reaction_v)
        reaction_int.append(reaction_i)

    return torch.tensor(reaction_vector, dtype=torch.float32), torch.tensor(reaction_int)


def gcn_attributes(gcn_ut_pairs, reaction_vector):
    edge_index = []
    edge_type = []
    for i in range(gcn_ut_pairs.size(1)):
        reaction_v = reaction_vector[i]
        seen = sum(reaction_v) == 0

        if seen:
            edge_i = torch.stack((gcn_ut_pairs[:, i],))
            edge_t = torch.tensor([0])
        else:
            edge_i = gcn_ut_pairs[:, i].repeat(int(sum(reaction_v)), 1)
            edge_t = torch.t(torch.nonzero(reaction_v))[0] + 1

        edge_index.extend(edge_i.tolist())
        edge_type.extend(edge_t.tolist())
    return torch.t(torch.tensor(edge_index)), torch.tensor(edge_type)
