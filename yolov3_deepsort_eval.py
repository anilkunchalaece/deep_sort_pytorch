import os
import os.path as osp
import logging
import argparse
from pathlib import Path

from utils.log import get_logger
from deepsort import VideoTracker, imgSeqTracker
from utils.parser import get_config

import motmetrics as mm
mm.lap.default_solver = 'lap'
from utils.evaluation import Evaluator

def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

def main(data_root='', seqs=('',), args=""):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    data_type = 'mot'
    result_root = os.path.join(Path(data_root), "mot_results")
    mkdir_if_missing(result_root)

    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    cfg.USE_FASTREID = False
    cfg.USE_MMDET = False
    cfg.USE_FSINET = False
    if cfg.USE_FSINET :
        cfg.merge_from_file(args.config_fsinet)
    else :
        attr_dir = None

    # run tracking
    accs = []
    for seq in seqs:
        print('start seq: {}'.format(seq))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        # video_path = data_root+"/"+seq+"/video/video.mp4"
        # src_dir = data_root+"/"+seq+"/img1"
        src_dir = data_root+"/"+seq+"/wireframes/pred/pare_results/"
        attr_dir = None
        cfg.USE_FSINET = False
        # attr_dir = data_root+"/"+seq+"/wireframes/metadata/"
        # src_dir = data_root+"/"+seq+"/wireframes/src/orig_images_scaled/"

        # with VideoTracker(cfg, args, video_path) as vdo_trk:
        #     vdo_trk.run()

        img_trk =  imgSeqTracker(cfg, args, src_dir,result_filename)
        img_trk.run(attr_dir)

        # eval
        print('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        print(F"results saved in {result_filename}")

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        cfg.USE_FSINET = True
        cfg.merge_from_file(args.config_fsinet)

        # run with FSINet
        result_filename = os.path.join(result_root, '{}_FSI_FUSED.txt'.format(seq))
        # video_path = data_root+"/"+seq+"/video/video.mp4"
        # src_dir = data_root+"/"+seq+"/img1"
        src_dir = data_root+"/"+seq+"/wireframes/pred/pare_results/"
        attr_dir = data_root+"/"+seq+"/wireframes/metadata/"
        # src_dir = data_root+"/"+seq+"/wireframes/src/orig_images_scaled/"

        # with VideoTracker(cfg, args, video_path) as vdo_trk:
        #     vdo_trk.run()

        img_trk =  imgSeqTracker(cfg, args, src_dir,result_filename)
        img_trk.run(attr_dir)

        print('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        print(F"results saved in {result_filename}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        cfg.USE_FSINET = True
        cfg.merge_from_file(args.config_fsinet_different_color_without_background)

        # run with FSINet
        result_filename = os.path.join(result_root, '{}_FSI_FUSED_DIFF_COLOR_WITHOUT_BG.txt'.format(seq))
        # video_path = data_root+"/"+seq+"/video/video.mp4"
        # src_dir = data_root+"/"+seq+"/img1"
        src_dir = data_root+"/"+seq+"/wireframes/pred/pare_results/"
        attr_dir = data_root+"/"+seq+"/wireframes/metadata/"
        # src_dir = data_root+"/"+seq+"/wireframes/src/orig_images_scaled/"

        # with VideoTracker(cfg, args, video_path) as vdo_trk:
        #     vdo_trk.run()

        img_trk =  imgSeqTracker(cfg, args, src_dir,result_filename)
        img_trk.run(attr_dir)

        # eval
        print('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        print(F"results saved in {result_filename}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        cfg.USE_FSINET = True
        cfg.merge_from_file(args.config_fsinet_different_color_with_background)

        # run with FSINet
        result_filename = os.path.join(result_root, '{}_FSI_FUSED_DIFF_COLOR_WITH_BG.txt'.format(seq))
        # video_path = data_root+"/"+seq+"/video/video.mp4"
        # src_dir = data_root+"/"+seq+"/img1"
        src_dir = data_root+"/"+seq+"/wireframes/pred/pare_results/"
        attr_dir = data_root+"/"+seq+"/wireframes/metadata/"
        # src_dir = data_root+"/"+seq+"/wireframes/src/orig_images_scaled/"

        # with VideoTracker(cfg, args, video_path) as vdo_trk:
        #     vdo_trk.run()

        img_trk =  imgSeqTracker(cfg, args, src_dir,result_filename)
        img_trk.run(attr_dir)

        # eval
        print('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        print(F"results saved in {result_filename}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        cfg.USE_FSINET = True
        cfg.merge_from_file(args.config_fsinet_img_only)
        # run with FSINet
        result_filename = os.path.join(result_root, '{}_FSI_IMG_ONLY.txt'.format(seq))
        # video_path = data_root+"/"+seq+"/video/video.mp4"
        # src_dir = data_root+"/"+seq+"/img1"
        src_dir = data_root+"/"+seq+"/wireframes/pred/pare_results/"
        attr_dir = data_root+"/"+seq+"/wireframes/metadata/"
        # src_dir = data_root+"/"+seq+"/wireframes/src/orig_images_scaled/"

        # with VideoTracker(cfg, args, video_path) as vdo_trk:
        #     vdo_trk.run()

        img_trk =  imgSeqTracker(cfg, args, src_dir,result_filename)
        img_trk.run(attr_dir)

        # eval
        print('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        print(F"results saved in {result_filename}")
        # return

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    new_seqs = []
    for s in seqs :
        new_seqs.extend([s, s+"fsi_FUSED" , s+"diff_color_without_bg", s+"diff_color_with_bg",s+"fsi_IMG_ONLY"])
        # new_seqs.extend([s, s+"fsi_FUSED" ])

    summary = Evaluator.get_summary(accs, new_seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_global.xlsx'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fsinet", type=str, default="./configs/fsinet.yaml")
    parser.add_argument("--config_fsinet_img_only", type=str, default="./configs/fsinet_img_only.yaml")
    parser.add_argument("--config_fsinet_different_color_without_background", type=str, default="./configs/fsinet_different_color_without_background.yaml")
    parser.add_argument("--config_fsinet_different_color_with_background", type=str, default="./configs/fsinet_different_color_with_background.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    seqs_str = '''MOT16-02       
                  MOT16-04
                  MOT16-05
                  MOT16-09
                  MOT16-10
                  MOT16-11
                  MOT16-13
                  '''     

    # seqs_str = '''
    #               MOT16-02
    #               MOT16-04       
    #               '''     


    data_root = '/home/akunchala/Documents/z_Datasets/MOT16/train'

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root,
         seqs=seqs,
         args=args)