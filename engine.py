import os
import sys
import pdb
import tqdm
import torch
import utils
import numpy as np
from copy import deepcopy
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datasets.coco_eval import CocoEvaluator
from transformers_local.models.deformable_detr.image_processing_deformable_detr import DeformableDetrImageProcessor
from transformers_local.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig
from transformers_local.models.deformable_detr.modeling_deformable_detr import DeformableDetrForObjectDetection
class local_trainer(pl.LightningModule):
	def __init__(self, train_loader, val_loader, test_dataset, args, local_evaluator, task_id):
		super().__init__()

		detr_config = DeformableDetrConfig()
		detr_config.num_labels = args.n_classes #+ 1
		detr_config.PREV_INTRODUCED_CLS = args.task_map[task_id][1]
		detr_config.CUR_INTRODUCED_CLS = args.task_map[task_id][2]
		seen_classes = detr_config.PREV_INTRODUCED_CLS + detr_config.CUR_INTRODUCED_CLS
		
		#### prompt arguments
		detr_config.use_prompts = args.use_prompts
		detr_config.n_tasks = args.n_tasks
		detr_config.num_prompts = args.num_prompts
		detr_config.prompt_len = args.prompt_len

		self.invalid_cls_logits = list(range(seen_classes, args.n_classes-1)) #unknown class indx will not be included in the invalid class range

		if args.repo_name:
			self.model =  DeformableDetrForObjectDetection.from_pretrained(args.repo_name,config=detr_config,
																	ignore_mismatched_sizes=True,
																	default=not(args.mask_gradients), log_file=args.log_file)
			self.processor = DeformableDetrImageProcessor.from_pretrained(args.repo_name)
		else:
			self.model = DeformableDetrForObjectDetection(detr_config, default=not(args.mask_gradients),
												 log_file=args.log_file)
			
			self.processor = DeformableDetrImageProcessor()
		#  self.model = DetrForObjectDetection.from_pretrained(
		#     pretrained_model_name_or_path=repo_name, 
		#     num_labels=args.n_classes,
		#     ignore_mismatched_sizes=True
		#  )
		# see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
		self.task_id = task_id
		self.lr = args.lr
		self.lr_backbone = args.lr_backbone
		self.weight_decay = args.weight_decay
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_dataset = test_dataset
		self.args = args
		#self.processor = DetrImageProcessor.from_pretrained(repo_name)
		self.print_count = 0
		self.evaluator = local_evaluator
		self.evaluator.model = self.model
		self.evaluator.invalid_cls_logits = self.invalid_cls_logits
		self.PREV_INTRODUCED_CLS = args.task_map[task_id][1]

		#self.automatic_optimization = False
	
	def forward(self, pixel_values, pixel_mask):
		outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
		return outputs

	def common_step(self, batch, batch_idx, return_outputs=None):
		pixel_values = batch["pixel_values"].to(self.device)
		pixel_mask = batch["pixel_mask"].to(self.device)
		labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

		if self.args.use_prompts:
			#pdb.set_trace()
			with torch.no_grad():
				outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, train=False, task_id=self.task_id)
				query = outputs.last_hidden_state.mean(dim=1)
		else:
			query = None

		outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, query=query, train=True, task_id=self.task_id)

		loss = outputs.loss
		loss_dict = outputs.loss_dict

		if return_outputs:
			orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
		
			if self.args.mask_gradients:
				outputs.logits[:,:, self.invalid_cls_logits] = -10e10
				outputs.logits = outputs.logits[:,:,:self.args.n_classes-1] #removing background class
		
			# TODO: fix  processor.post_process_object_detection()
			results = self.processor.post_process(outputs, target_sizes=orig_target_sizes) # convert outputs to COCO api
			res = {target['image_id'].item(): output for target, output in zip(labels, results)}
			res = self.evaluator.prepare_for_coco_detection(res)
		
			return loss, loss_dict, res

		return loss, loss_dict
	
	def training_step(self, batch, batch_idx): # automatic training schedule

		loss, loss_dict = self.common_step(batch, batch_idx)
		# logs metrics for each training_step,
		# and the average across the epoch
		#values = {k:v for k,v in loss_dict.items()}
		short_map = {'loss_ce':'ce','loss_giou':'giou','cardinality_error':'car','training_loss':'tr','loss_bbox':'bbox'}
		self.log("tr", loss, prog_bar=True)
		for k,v in loss_dict.items():
			#self.log("train_" + k, v.item(), prog_bar=True)
			self.log(short_map[k], v.item(), prog_bar=True)

		return loss

	def on_after_backward(self, *args):
		# pdb.set_trace()
		# for i in range(len(self.model.class_embed)):
		# 	self.model.class_embed[i].weight.grad[:self.PREV_INTRODUCED_CLS,:] = 0
		# 	self.model.class_embed[i].bias.grad[:self.PREV_INTRODUCED_CLS] = 0
		# print ('begin',self.model.class_embed[0].weight[:self.PREV_INTRODUCED_CLS,:].norm())
		# print ('begin',self.model.class_embed[0].weight[self.PREV_INTRODUCED_CLS:,:].norm())
		return
	
	# def on_train_batch_end(self, *args): # runs after optimizer.step()
	# 	pdb.set_trace()
	# 	print ('end',self.model.class_embed[0].weight[:self.PREV_INTRODUCED_CLS,:].norm())

	# def training_step(self, batch, batch_idx): # manual training schedule
	# 	opt = self.optimizers()
	# 	#opt.zero_grad()	
	# 	loss, loss_dict = self.common_step(batch, batch_idx)
	# 	# logs metrics for each training_step,
	# 	# and the average across the epoch
	# 	#values = {k:v for k,v in loss_dict.items()}
	# 	short_map = {'loss_ce':'ce','loss_giou':'giou','cardinality_error':'car','training_loss':'tr','loss_bbox':'bbox'}
	# 	self.log("tr", loss, prog_bar=True)
	# 	for k,v in loss_dict.items():
	# 		#self.log("train_" + k, v.item(), prog_bar=True)
	# 		self.log(short_map[k], v.item(), prog_bar=True)
	# 	#pdb.set_trace()
		
	# 	self.manual_backward(loss)

	# 	for i in range(len(self.model.class_embed)):
	# 		self.model.class_embed[i].weight.grad[:self.PREV_INTRODUCED_CLS,:] = 0
	# 		self.model.class_embed[i].bias.grad[:self.PREV_INTRODUCED_CLS] = 0
	# 	#pdb.set_trace()
	# 	self.clip_gradients(opt, gradient_clip_val=0.1, gradient_clip_algorithm="norm")

	# 	if (batch_idx + 1) % 32 == 0 or batch_idx == self.trainer.num_train_batches[0]-1:
	# 		opt.step()
	# 		opt.zero_grad()
	# 	#pdb.set_trace()
	# 	return loss

	def on_train_epoch_end(self):
		self.lr_scheduler.step()
		if self.current_epoch and self.current_epoch%self.args.save_epochs == 0:
			self.save(self.current_epoch)

	def validation_step(self, batch, batch_idx):
		#print (batch_idx)
		if batch_idx == 0:
			self.coco_evaluator = CocoEvaluator(self.test_dataset.coco, self.args.iou_types)

		loss, loss_dict, res = self.common_step(batch, batch_idx, return_outputs=True)
		self.coco_evaluator.update(res)

		#print ('\nhere', batch_idx, self.trainer.num_val_batches, '\n')

		self.log("validation_loss", loss, sync_dist=True)
		for k,v in loss_dict.items():
			self.log("validation_" + k, v.item(), sync_dist=True)

		if batch_idx == self.trainer.num_val_batches[0]-1:
		#if True:
			#self.save(self.current_epoch)
			self.coco_evaluator.synchronize_between_processes()
			self.coco_evaluator.accumulate()
			self.coco_evaluator.summarize()
			stats = self.coco_evaluator.coco_eval[self.args.iou_types[0]].stats

			if self.trainer.global_rank == 0:
				self.evaluator.print_coco_stats(self.current_epoch, stats, self.print_count)
				self.print_count = 1
				if self.args.viz:
					image_ids = self.test_dataset.coco.getImgIds()
					#print(image_ids[0:4])
					for id in image_ids[0:self.args.num_imgs_viz]:
						self.evaluator.vizualize(id=id)
			# with open(self.default_root_dir+'/stats.txt','a') as f:
			#     f.write()

		return loss
	
	def save(self, epoch):
		print('\n Saving at epoch ', epoch, file=self.args.log_file)
		torch.save({
					'model': self.model.state_dict(),
					'optimizer': self.optimizer.state_dict(),
					'lr_scheduler': self.lr_scheduler.state_dict(),
					'epoch': epoch,
					#'args': self.args,
				}, os.path.join(self.args.output_dir, f'checkpoint{epoch:02}.pth'))
	
	def resume(self, load_path=''):
		print('\n Resuming model for task ', self.task_id, ' from : ',load_path, file=self.args.log_file)
		if load_path:
			checkpoint = torch.load(load_path, map_location='cpu')
			missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model'], strict=False)

		if not self.args.eval and self.args.freeze:
			
			freeze = self.args.freeze.split(',')
			for id, (name, params) in enumerate(self.model.named_parameters()):
				params.requires_grad = True
				flag = False
				for n in name.split('.'):
					# if n == 'project_prompts':
					# 	params.requires_grad = True
					# 	flag = False
					# 	break
					if n in freeze:
						params.requires_grad = False
						flag = True
				if not flag:
					#if self.trainer.global_rank == 0:
					print ('Trainable ..', name, "  Req grad .. ",params.requires_grad, file=self.args.log_file)
		
		# unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
		# # if len(missing_keys) > 0:
		# # 	print('Missing Keys: {}'.format(missing_keys))
		# if len(unexpected_keys) > 0:
		# 	print('Unexpected Keys: {}'.format(unexpected_keys))
	
		# if optimizer:
		# 	p_groups = copy.deepcopy(optimizer.param_groups)
		#     optimizer.load_state_dict(model['optimizer'])
		#     for pg, pg_old in zip(optimizer.param_groups, p_groups):
		#         pg['lr'] = pg_old['lr']
		#         pg['initial_lr'] = pg_old['initial_lr']
	
	def match_name_keywords(self, n, name_keywords):
		out = False
		for b in name_keywords:
			if b in n:
				out = True
				break
		return out
	
	def configure_optimizers(self):
		new_params = self.args.new_params.split(',')

		if self.args.repo_name:
			param_dicts = [
				{"params": [p for n, p in self.named_parameters()
					if self.match_name_keywords(n, new_params) and p.requires_grad],
					"lr":self.args.lr,
					},
				{
					"params": [p for n, p in self.named_parameters() if not self.match_name_keywords(n, new_params) and p.requires_grad],
					"lr": self.args.lr_old,
				},
			]
		else:
			param_dicts = [
			{
				"params":
					[p for n, p in self.named_parameters()
					if not self.match_name_keywords(n, self.args.lr_backbone_names) and not self.match_name_keywords(n, self.args.lr_linear_proj_names) and p.requires_grad],
				"lr": self.args.lr,
			},
			{
				"params": [p for n, p in self.named_parameters() if self.match_name_keywords(n, self.args.lr_backbone_names) and p.requires_grad],
				"lr": self.args.lr_backbone,
			},
			{
				"params": [p for n, p in self.named_parameters() if self.match_name_keywords(n, self.args.lr_linear_proj_names) and p.requires_grad],
				"lr": self.args.lr * self.args.lr_linear_proj_mult,
			}
			]

		self.optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
								weight_decay=self.weight_decay)

		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_drop)

		return self.optimizer

	def train_dataloader(self):
		return self.train_dataloader

	def val_dataloader(self):
		return self.val_dataloader

class Evaluator():
	def __init__(self, processor, test_dataset, test_dataloader, coco_evaluator, 
			  task_label2name, args, local_trainer=None, PREV_INTRODUCED_CLS=0, 
			  CUR_INTRODUCED_CLS=20, local_eval=0, task_id=0):
		
		self.processor = processor
		self.local_trainer = local_trainer
		if local_trainer:
			self.model = local_trainer.model
		else:
			self.model = None

		self.test_dataset = test_dataset
		self.test_dataloader = test_dataloader
		self.coco_evaluator = coco_evaluator
		self.task_label2name = task_label2name
		self.args = args
		self.local_eval = local_eval
		self.task_id = task_id

		#if self.args.mask_gradients:
		prev_intro_cls = PREV_INTRODUCED_CLS
		curr_intro_cls = CUR_INTRODUCED_CLS
		seen_classes = prev_intro_cls + curr_intro_cls
		#self.invalid_cls_logits = list(range(seen_classes, self.args.n_classes-1)) #unknown class indx will not be included in the invalid class range
		self.invalid_cls_logits = list(range(seen_classes, self.args.n_classes-1)) #unknown class indx will not be included in the invalid class range

	def convert_to_xywh(self, boxes):
		xmin, ymin, xmax, ymax = boxes.unbind(1)
		return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

	def prepare_for_coco_detection(self, predictions):
		coco_results = []
		for original_id, prediction in predictions.items():
			if len(prediction) == 0:
				continue

			boxes = prediction["boxes"]
			boxes = self.convert_to_xywh(boxes).tolist()
			scores = prediction["scores"].tolist()
			labels = prediction["labels"].tolist()

			coco_results.extend(
				[
					{
						"image_id": original_id,
						"category_id": labels[k],
						"bbox": box,
						"score": scores[k],
					}
					for k, box in enumerate(boxes)
				]
			)
		return coco_results
	
	def print_coco_stats(self, epoch, stats, print_count):
		#print(stats)
		# output = []
		# if self.args.switch:
		# 	sep = "------------------------------------------------------------------------------- \n------------------------------------------------------------------------------- \n"
		# else:
		# 	sep=''

		output = ['Task '+self.args.task,
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '+'%0.2f'%stats[0],
		'\nAverage Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = '+'%0.2f'%stats[1],
		'\nAverage Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = '+'%0.2f'%stats[2],
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = '+'%0.2f'%stats[3],
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = '+'%0.2f'%stats[4],
		'\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = '+'%0.2f'%stats[5],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = '+'%0.2f'%stats[6],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = '+'%0.2f'%stats[7],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '+'%0.2f'%stats[8],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = '+'%0.2f'%stats[9],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = '+'%0.2f'%stats[10],
		'\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = '+'%0.2f'%stats[11],
		'\n\n']
		
		if print_count == 0:
			print_format = 'w'
		else:
			print_format = 'a'
		with open(self.args.output_dir+'/stats.txt', print_format) as f:
			f.writelines(output)
		f.close()
	
	def plot_results(self, pil_img, ax, scores, labels, boxes):
		COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
		[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
		#plt.figure(figsize=(16,10))
		ax.imshow(pil_img)
		#ax = plt.gca()
		colors = COLORS * 100
		for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
			ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
									fill=False, color=c, linewidth=2))
			text = f'{self.task_label2name[label]}: {score:0.2f}'
			ax.text(xmin, ymin, text, fontsize=5,
					bbox=dict(facecolor='yellow', alpha=0.5))
		ax.grid('off')

	def evaluate(self):
		args = self.args
		model = self.model
		coco_evaluator = self.coco_evaluator
		print("\n Running Final Evaluation... \n", file=self.args.log_file)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		model.eval()

		for idx, batch in enumerate(tqdm.tqdm(self.test_dataloader)):
			
			pixel_values = batch["pixel_values"].to(device)
			pixel_mask = batch["pixel_mask"].to(device)
			labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

			# # forward pass
			# with torch.no_grad():
			# 	outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

			if self.args.use_prompts:
				with torch.no_grad():
					outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, train=False)
					query = outputs.last_hidden_state.mean(dim=1)
			else:
				query = None

			outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, query=query, train=False)

			if self.args.mask_gradients:
				outputs.logits[:,:, self.invalid_cls_logits] = -10e10
				outputs.logits = outputs.logits[:,:,:self.args.n_classes-1] #removing background class
	
			orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
			results = self.processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes,
															threshold=0) # convert outputs to COCO api
			res = {target['image_id'].item(): output for target, output in zip(labels, results)}
			res = self.prepare_for_coco_detection(res)
			coco_evaluator.update(res)

		coco_evaluator.synchronize_between_processes()
		coco_evaluator.accumulate()
		coco_evaluator.summarize()

		#if not self.local_trainer or self.local_trainer.trainer.global_rank == 0:
		if self.local_eval:
			self.print_coco_stats(epoch=args.epochs+1, stats=coco_evaluator.coco_eval[args.iou_types[0]].stats, print_count=1)
		elif self.local_trainer.trainer.global_rank == 0:
			self.print_coco_stats(epoch=args.epochs+1, stats=coco_evaluator.coco_eval[args.iou_types[0]].stats, print_count=1)

		if args.viz:
			image_ids = self.test_dataset.coco.getImgIds()
			#print(image_ids[0:4])
			for id in image_ids[0:self.args.num_imgs_viz]:
				self.vizualize()
	# def evaluate_allTasks(self):


	def vizualize(self, id=None, score_threshold=0.3, device='cuda'):
		test_dataset = self.test_dataset
		image_ids = test_dataset.coco.getImgIds()
		
		if id == None:
			image_id = image_ids[np.random.randint(0, len(image_ids))]
		else:
			image_id = id
		print('Image n°{}'.format(image_id))
		image = test_dataset.coco.loadImgs(image_id)[0]
		image = Image.open(os.path.join(self.args.test_img_dir, image['file_name']))

		fig, ax = plt.subplots(1, 2, figsize=(14,6), dpi=220)

		# plotting GT
		annotations = test_dataset.coco.imgToAnns[image_id]
		cats = test_dataset.coco.cats
		id2label = {k: v['name'] for k,v in cats.items()}
		scores, labels, boxes = [],[],[]
		for annotation in annotations:
			box = annotation['bbox']
			class_idx = annotation['category_id']
			x,y,w,h = tuple(box)
			scores.append(1.0)
			labels.append(class_idx)
			boxes.append((x,y,x+w,y+h))
			
		ax[0].set_title('GT')
		self.plot_results(image,ax=ax[0],scores=np.array(scores),labels=np.array(labels),boxes=np.array(boxes))

		# plotting model's inference
		inputs = self.processor(images=image, return_tensors="pt")
		#pixel_values = inputs["pixel_values"].squeeze()
		inputs['pixel_values'] = inputs['pixel_values'].to(device)
		inputs['pixel_mask'] = inputs['pixel_mask'].to(device)

		# with torch.no_grad():
		# 	outputs = self.model(**inputs)

		if self.args.use_prompts:
			#pdb.set_trace()
			with torch.no_grad():
				outputs = self.model(pixel_values=inputs['pixel_values'] , pixel_mask=inputs['pixel_mask'], train=False)
				query = outputs.last_hidden_state.mean(dim=1)
		else:
			query = None

		#pdb.set_trace()
		with torch.no_grad():
			outputs = self.model(pixel_values=inputs['pixel_values'] , pixel_mask=inputs['pixel_mask'], query=query, train=False)

		if self.args.mask_gradients:
			outputs.logits[:,:, self.invalid_cls_logits] = -10e10
			outputs.logits = outputs.logits[:,:,:self.args.n_classes-1] #removing background class
		# let's only keep predictions with score > 0.3
		#pdb.set_trace()
			
		task = 'cur'
		if len(self.args.task) > 1:
			task = 'prev'
			score_threshold = 0.14
			
		results = self.processor.post_process_object_detection(outputs,target_sizes=[image.size[::-1]],
															threshold=score_threshold)[0]

		ax[1].set_title('Prediction (Ours)')
		self.plot_results(image,ax=ax[1],scores=results['scores'],labels=results['labels'],boxes=results['boxes'])
		
		for i in range(2):
				ax[i].set_aspect('equal')
				ax[i].set_axis_off()

		plt.savefig(os.path.join(self.args.output_dir, f'{task}_img_{image_id}.jpg'), bbox_inches = 'tight',pad_inches = 0.1)