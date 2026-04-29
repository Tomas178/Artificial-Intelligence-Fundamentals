import os
import time

from consts import DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY
from DataReader import ImageDataReader, KeystrokeDataReader
from Enums import ActivationFunction, Optimizer
from Models import ImageCNN, KeystrokeCNN
from torch.utils.data import DataLoader
from Utils import KeystrokeDataset, Trainer, get_device

RESULTS_DIRECTORY = 'results'

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

IMAGE_NUM_CLASSES = 2
KEYSTROKE_NUM_CLASSES = 30

IMAGE_SWEEP_SUBSET_RATIO = 0.2
IMAGE_EVAL_SUBSET_RATIO = 0.5

DOMAIN_IMAGE = 'image'
DOMAIN_KEYSTROKE = 'keystroke'


# Vaizdinių Duomenų užkrovimas
def load_image_data(subset_ratio=IMAGE_EVAL_SUBSET_RATIO):
	image_directory = os.path.join(DATASETS_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY)
	reader = ImageDataReader(image_directory, subset_ratio=subset_ratio)
	train_set, val_set, test_set = reader.read()

	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
	test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

	return train_loader, val_loader, test_loader, train_set.classes


# Laiko Eilučių duomenų užkrovimas
def load_keystroke_data():
	keystroke_directory = os.path.join(DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY)
	reader = KeystrokeDataReader(keystroke_directory)
	train_df, val_df, test_df = reader.read()

	train_dataset = KeystrokeDataset(train_df)
	val_dataset = KeystrokeDataset(val_df, label_map=train_dataset.label_map)
	test_dataset = KeystrokeDataset(test_df, label_map=train_dataset.label_map)

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

	return train_loader, val_loader, test_loader, train_dataset.classes


# Eksperimento vykdymas
def run_experiment(
	model,
	train_loader,
	val_loader,
	test_loader,
	save_directory,
	display_name,
	device,
	optimizer=Optimizer.ADAM,
	class_names=None,
):
	print(f'\n=== {display_name} ===')

	trainer = Trainer(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		test_loader=test_loader,
		optimizer=optimizer,
		learning_rate=LEARNING_RATE,
		device=device,
	)

	trainer.train(epochs=EPOCHS)
	test_loss, test_accuracy = trainer.evaluate()

	os.makedirs(save_directory, exist_ok=True)

	trainer.plot_history(
		title=display_name,
		save_path=os.path.join(save_directory, 'history.png'),
	)

	confusion_mat = trainer.get_confusion_matrix()
	Trainer.plot_confusion_matrix(
		confusion_mat,
		class_names=class_names,
		title=f'{display_name} - Confusion Matrix',
		save_path=os.path.join(save_directory, 'confusion_matrix.png'),
	)

	return trainer.history, test_loss, test_accuracy


def _run_specs(specs, loaders, study_dir, study_display, domain, device):
	train, val, test, classes = loaders
	histories = {}
	outcomes = []
	study_root = os.path.join(RESULTS_DIRECTORY, study_dir)
	for label, factory, run_kwargs in specs:
		exp_slug = f'{domain}_{label}'.replace(' ', '_').lower()
		save_directory = os.path.join(study_root, exp_slug)
		display_name = f'{study_display} - {domain.title()} - {label}'
		model = factory()
		history, test_loss, test_accuracy = run_experiment(
			model,
			train,
			val,
			test,
			save_directory,
			display_name,
			device,
			class_names=classes,
			**run_kwargs,
		)
		histories[label] = history
		outcomes.append(
			{
				'label': display_name,
				'variant': label,
				'study': study_display,
				'study_dir': study_dir,
				'domain': domain,
				'factory': factory,
				'run_kwargs': run_kwargs,
				'history': history,
				'test_loss': test_loss,
				'test_accuracy': test_accuracy,
			}
		)
	return histories, outcomes


def run_study(
	study_dir,
	study_display,
	image_specs,
	keystroke_specs,
	image_loaders,
	keystrokes_loaders,
	device,
):
	print('\n' + '=' * 60)
	print(study_display)
	print('=' * 60)

	study_start = time.perf_counter()
	study_root = os.path.join(RESULTS_DIRECTORY, study_dir)
	os.makedirs(study_root, exist_ok=True)

	image_histories, image_outcomes = _run_specs(
		image_specs,
		image_loaders,
		study_dir,
		study_display,
		DOMAIN_IMAGE,
		device,
	)
	Trainer.plot_comparison(
		image_histories,
		f'Images - {study_display}',
		os.path.join(study_root, 'img_comparison.png'),
	)

	keystroke_histories, keystroke_outcomes = _run_specs(
		keystroke_specs,
		keystrokes_loaders,
		study_dir,
		study_display,
		DOMAIN_KEYSTROKE,
		device,
	)
	Trainer.plot_comparison(
		keystroke_histories,
		f'Keystrokes - {study_display}',
		os.path.join(study_root, 'ks_comparison.png'),
	)

	elapsed = time.perf_counter() - study_start
	print(f'\n[Study time: {elapsed:.1f}s ({elapsed / 60:.2f} min)]')

	return image_outcomes + keystroke_outcomes


# Tyrimas 1: Architektūros palyginimas
def run_architecture_experiments(image_loaders, keystrokes_loaders, device):
	image_architectures = {
		'Small (16,32)': (16, 32),
		'Medium (32,64,128)': (32, 64, 128),
		'Large (64,128,256)': (64, 128, 256),
	}
	keystroke_architectures = {
		'Small (32,)': (32,),
		'Medium (64,128)': (64, 128),
		'Large (64,128,256)': (64, 128, 256),
	}
	image_specs = [
		(name, lambda f=filters: ImageCNN(num_classes=IMAGE_NUM_CLASSES, num_filters=f), {})
		for name, filters in image_architectures.items()
	]
	keystroke_specs = [
		(
			name,
			lambda f=filters: KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES, num_filters=f),
			{},
		)
		for name, filters in keystroke_architectures.items()
	]
	return run_study(
		'architecture',
		'STUDY 1: Architecture Comparison',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
	)


# Tyrimas 2: Dropout palyginimas
def run_dropout_experiments(image_loaders, keystrokes_loaders, device):
	dropout_values = [0.0, 0.25, 0.5, 0.75]
	image_specs = [
		(f'Dropout={d}', lambda d=d: ImageCNN(num_classes=IMAGE_NUM_CLASSES, dropout=d), {})
		for d in dropout_values
	]
	keystroke_specs = [
		(
			f'Dropout={d}',
			lambda d=d: KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES, dropout=d),
			{},
		)
		for d in dropout_values
	]
	return run_study(
		'dropout',
		'STUDY 2: Dropout Comparison',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
	)


# Tyrimas 3: Batch Normalization
def run_batch_norm_experiments(image_loaders, keystrokes_loaders, device):
	bn_options = [(False, 'Without BN'), (True, 'With BN')]
	image_specs = [
		(
			label,
			lambda use_bn=use_bn: ImageCNN(num_classes=IMAGE_NUM_CLASSES, use_batch_norm=use_bn),
			{},
		)
		for use_bn, label in bn_options
	]
	keystroke_specs = [
		(
			label,
			lambda use_bn=use_bn: KeystrokeCNN(
				num_classes=KEYSTROKE_NUM_CLASSES, use_batch_norm=use_bn
			),
			{},
		)
		for use_bn, label in bn_options
	]
	return run_study(
		'batch_norm',
		'STUDY 3: Batch Normalization',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
	)


# Tyrimas 4: Aktyvacijos funkcijos
def run_activation_experiments(image_loaders, keystrokes_loaders, device):
	activations = [
		ActivationFunction.RELU,
		ActivationFunction.LEAKY_RELU,
		ActivationFunction.TANH,
		ActivationFunction.ELU,
	]
	image_specs = [
		(
			str(activation),
			lambda a=activation: ImageCNN(num_classes=IMAGE_NUM_CLASSES, activation=a),
			{},
		)
		for activation in activations
	]
	keystroke_specs = [
		(
			str(activation),
			lambda a=activation: KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES, activation=a),
			{},
		)
		for activation in activations
	]
	return run_study(
		'activation',
		'STUDY 4: Activation Functions',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
	)


# Tyrimas 5: Optimizavimo algoritmai
def run_optimizer_experiments(image_loaders, keystrokes_loaders, device):
	image_specs = [
		(
			str(optimizer),
			lambda: ImageCNN(num_classes=IMAGE_NUM_CLASSES),
			{'optimizer': optimizer},
		)
		for optimizer in Optimizer
	]
	keystroke_specs = [
		(
			str(optimizer),
			lambda: KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES),
			{'optimizer': optimizer},
		)
		for optimizer in Optimizer
	]
	return run_study(
		'optimizer',
		'STUDY 5: Optimizers',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
	)


def best_epoch_summary(history):
	val_acc = history['val_acc']
	val_loss = history['val_loss']
	best_idx = max(range(len(val_acc)), key=lambda i: (val_acc[i], -val_loss[i]))
	return {
		'epoch': best_idx + 1,
		'val_acc': val_acc[best_idx],
		'val_loss': val_loss[best_idx],
		'train_acc': history['train_acc'][best_idx],
		'train_loss': history['train_loss'][best_idx],
	}


def pick_best(outcomes):
	def score(outcome):
		summary = best_epoch_summary(outcome['history'])
		return (summary['val_acc'], -summary['val_loss'])

	return max(outcomes, key=score)


def evaluate_best_model(outcome, loaders, device, dir_name, display_name, title_name):
	summary = best_epoch_summary(outcome['history'])
	print(f'\n--- Best {display_name} Model ---')
	print(f'Selected from: {outcome["study"]}')
	print(f'Variant: {outcome["variant"]}')
	print(
		f'Sweep best epoch={summary["epoch"]} | '
		f'val_acc={summary["val_acc"]:.4f} | val_loss={summary["val_loss"]:.4f} | '
		f'train_acc={summary["train_acc"]:.4f} | train_loss={summary["train_loss"]:.4f}'
	)
	print(f'Sweep test_acc={outcome["test_accuracy"]:.4f} | test_loss={outcome["test_loss"]:.4f}')
	print('Retraining with larger dataset...\n')

	train, val, test, classes = loaders
	model = outcome['factory']()
	optimizer = outcome['run_kwargs'].get('optimizer', Optimizer.ADAM)

	trainer = Trainer(
		model=model,
		train_loader=train,
		val_loader=val,
		test_loader=test,
		optimizer=optimizer,
		learning_rate=LEARNING_RATE,
		device=device,
	)
	trainer.train(epochs=EPOCHS)
	final_test_loss, final_test_accuracy = trainer.evaluate()

	output_dir = os.path.join(RESULTS_DIRECTORY, 'best_model', dir_name)
	os.makedirs(output_dir, exist_ok=True)

	trainer.plot_history(
		title=f'Best {title_name} Model',
		save_path=os.path.join(output_dir, 'history.png'),
	)
	Trainer.plot_confusion_matrix(
		trainer.get_confusion_matrix(),
		class_names=classes,
		title=f'Best {title_name} Model - Confusion Matrix',
		save_path=os.path.join(output_dir, 'confusion_matrix.png'),
	)

	print(f'\n30 sample predictions ({display_name}):')
	for sample in trainer.get_sample_predictions(num_samples=30, class_names=classes):
		is_correct = '✓' if sample['true'] == sample['predicted'] else '✗'
		print(f'  {is_correct} True: {sample["true"]}, Predicted: {sample["predicted"]}')

	return final_test_loss, final_test_accuracy


# Punktas 4: Geriausio modelio įvertinimas
def run_best_model_evaluation(
	image_outcomes, keystroke_outcomes, image_loaders, keystroke_loaders, device
):
	print('\n' + '=' * 60)
	print('BEST MODEL EVALUATION')
	print('=' * 60)

	best_image = pick_best(image_outcomes)
	best_keystroke = pick_best(keystroke_outcomes)

	evaluate_best_model(best_image, image_loaders, device, 'image', 'image', 'Image')
	evaluate_best_model(
		best_keystroke,
		keystroke_loaders,
		device,
		'keystroke',
		'keystroke',
		'Keystroke',
	)


def _print_leaderboard(domain_label, outcomes):
	print(f'\n--- {domain_label} sweep summary (best epoch val_acc) ---')
	ranked = sorted(
		outcomes,
		key=lambda o: best_epoch_summary(o['history'])['val_acc'],
		reverse=True,
	)
	for outcome in ranked:
		summary = best_epoch_summary(outcome['history'])
		print(
			f'  val_acc={summary["val_acc"]:.4f} | '
			f'val_loss={summary["val_loss"]:.4f} | '
			f'test_acc={outcome["test_accuracy"]:.4f} | '
			f'{outcome["label"]}'
		)


def main():
	overall_start = time.perf_counter()

	device = get_device()
	os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

	# Užkrauname duomenis
	image_sweep_loaders = load_image_data(subset_ratio=IMAGE_SWEEP_SUBSET_RATIO)
	keystrokes_loaders = load_keystroke_data()

	all_outcomes = []
	all_outcomes.extend(
		run_architecture_experiments(image_sweep_loaders, keystrokes_loaders, device)
	)
	all_outcomes.extend(run_dropout_experiments(image_sweep_loaders, keystrokes_loaders, device))
	all_outcomes.extend(run_batch_norm_experiments(image_sweep_loaders, keystrokes_loaders, device))
	all_outcomes.extend(run_activation_experiments(image_sweep_loaders, keystrokes_loaders, device))
	all_outcomes.extend(run_optimizer_experiments(image_sweep_loaders, keystrokes_loaders, device))

	image_outcomes = [o for o in all_outcomes if o['domain'] == DOMAIN_IMAGE]
	keystroke_outcomes = [o for o in all_outcomes if o['domain'] == DOMAIN_KEYSTROKE]

	_print_leaderboard('Image', image_outcomes)
	_print_leaderboard('Keystroke', keystroke_outcomes)

	image_eval_loaders = load_image_data(subset_ratio=IMAGE_EVAL_SUBSET_RATIO)
	run_best_model_evaluation(
		image_outcomes,
		keystroke_outcomes,
		image_eval_loaders,
		keystrokes_loaders,
		device,
	)

	elapsed = time.perf_counter() - overall_start
	print('\n' + '=' * 60)
	print(f'TOTAL TIME: {elapsed:.1f} s ({elapsed / 60:.2f} min, {elapsed / 3600:.2f} h)')
	print('=' * 60)


if __name__ == '__main__':
	main()
