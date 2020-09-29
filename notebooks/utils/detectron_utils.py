"""
Interface  to Detectron2 
"""
from detectron2.config import get_cfg

def setup_cfg(hpc, ROOT_DIR):
    """
    Define Classifier for Panoptic Segmentation
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = ROOT_DIR + "/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x_model.pkl"
    if not hpc: 
        cfg.MODEL.DEVICE = 'cpu' #only on local device

    cfg.freeze()

    cfg = setup_cfg() #Define Classifier for Panoptic Segmentation
    predictor = DefaultPredictor(cfg)
    demo = VisualizationDemo(cfg)
    metadata = demo.metadata
    return cfg

def draw_panoptic_seg_predictions(
        self, frame, panoptic_seg, segments_info, area_threshold=None, alpha=0.5
    ):
        frame_visualizer = Visualizer(frame, self.metadata)
        pred = _PanopticPrediction(panoptic_seg, segments_info)

        if self._instance_mode == ColorMode.IMAGE_BW:
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                pred.non_empty_mask()
            )

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            frame_visualizer.draw_binary_mask(
                mask,
                color=mask_color,
                text=self.metadata.stuff_classes[category_idx],
                alpha=alpha,
                area_threshold=area_threshold,
            )

        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return frame_visualizer.output
        # draw mask for all instances second
        masks, sinfo = list(zip(*all_instances))
        num_instances = len(masks)
        masks_rles = mask_util.encode(
            np.asarray(np.asarray(masks).transpose(1, 2, 0), dtype=np.uint8, order="F")
        )
        assert len(masks_rles) == num_instances

        category_ids = [x["category_id"] for x in sinfo]
        detected = [
            _DetectedInstance(category_ids[i], bbox=None, mask_rle=masks_rles[i], color=None, ttl=8)
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)
        labels = [self.metadata.thing_classes[k] for k in category_ids]

        frame_visualizer.overlay_instances(
            boxes=None,
            masks=masks,
            labels=labels,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return frame_visualizer.output