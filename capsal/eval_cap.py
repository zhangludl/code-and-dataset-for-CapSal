from mrcnn.coco.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from mrcnn.coco.pycocoevalcap.bleu.bleu import Bleu
from mrcnn.coco.pycocoevalcap.meteor.meteor import Meteor
from mrcnn.coco.pycocoevalcap.rouge.rouge import Rouge
from mrcnn.coco.pycocoevalcap.cider.cider import Cider
# Down load the coco evaluation tool at https://pan.baidu.com/s/1mRN_qV7X8ZLUeuARQY3EwQ

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes


    def evaluate(self,imgIds):

        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = [k.encode('utf-8') for k in self.coco[str(imgId)]]
            res[imgId] = self.cocoRes[imgId]
        # imgIds = self.params['image_id']
        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        # tokenizer = PTBTokenizer()
        #
        # res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
