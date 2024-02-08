# Intra Shot MMR Multi Video summarization

In this project we introduces a novel adaptation of the MMR algorithm, Specifically, the summarization process is modified to include intra-shot dynamics. Our modified algorithm, which we will now refer to as Intra-Shot MMR (IS-MMR), adapts the classic MMR's diversity component. IS-MMR measures the uniqueness of frames within their original shots rather than the diversity of the full video or a chosen batch. The algorithm can now highlight significant moments and shot transitions due to this small but effective change, making the final summary both short and logical from a narrative standpoint. In the following equation:

$$
\text{IS-MMR}(f_i) = \lambda \text{Sim}_{\text{1}} (f_i, V \setminus C) - (1 - \lambda) \max_{h \in C} \text{Sim}_{\text{2}}(f_i, h)
$$
