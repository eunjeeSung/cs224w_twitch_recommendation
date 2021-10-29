def print_scores(scores):
    for k in ['all','new','rep']:
        print('{}: h@1: {:.5f} h@5: {:.5f} h@10: {:.5f} ndcg@1: {:.5f} ndcg@5: {:.5f} ndcg@10: {:.5f}'.format(
                                                        k,
                                                        scores[k]['h01'],
                                                        scores[k]['h05'],
                                                        scores[k]['h10'],
                                                        scores[k]['ndcg01'],
                                                        scores[k]['ndcg05'],
                                                        scores[k]['ndcg10'],
                                                      ))
    print("ratio: ", scores['ratio'])
