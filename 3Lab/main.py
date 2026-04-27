import os

from consts import DATASETS_DIRECTORY, KEYSTROKES_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY
from DataReader import ImageDataReader, KeystrokeDataReader
from Enums import ActivationFunction, Optimizer
from Models import ImageCNN, KeystrokeCNN
from torch.utils.data import DataLoader
from Utils import KeystrokeDataset, Trainer, get_device

RESULTS_DIRECTORY = 'results'

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.01

IMAGE_NUM_CLASSES = 2
KEYSTROKE_NUM_CLASSES = 30


# Vaizdinių Duomenų užkrovimas
def load_image_data():
	image_directory = os.path.join(DATASETS_DIRECTORY, MUFFIN_VS_CHIHUAHUA_DIRECTORY)
	reader = ImageDataReader(image_directory)
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
	experiment_name,
	device,
	optimizer=Optimizer.ADAM,
	class_names=None,
):
	print(f'\n=== {experiment_name} ===')

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

	save_directory = os.path.join(RESULTS_DIRECTORY, experiment_name.replace(' ', '_').lower())
	os.makedirs(save_directory, exist_ok=True)

	trainer.plot_history(
		title=experiment_name,
		save_path=os.path.join(save_directory, 'history.png'),
	)

	confusion_mat = trainer.get_confusion_matrix()
	Trainer.plot_confusion_matrix(
		confusion_mat,
		class_names=class_names,
		title=f'{experiment_name} - Confusion Matrix',
		save_path=os.path.join(save_directory, 'confusion_matrix.png'),
	)

	return trainer.history, test_loss, test_accuracy


def _run_specs(specs, loaders, label_prefix, device):
	train, val, test, classes = loaders
	results = {}
	for label, model, run_kwargs in specs:
		history, _, _ = run_experiment(
			model,
			train,
			val,
			test,
			f'{label_prefix} - {label}',
			device,
			class_names=classes,
			**run_kwargs,
		)
		results[label] = history
	return results


def run_study(
	study_name,
	image_specs,
	keystroke_specs,
	image_loaders,
	keystrokes_loaders,
	device,
	image_save,
	keystroke_save,
):
	print('\n' + '=' * 60)
	print(study_name)
	print('=' * 60)

	image_results = _run_specs(image_specs, image_loaders, f'Image {study_name}', device)
	Trainer.plot_comparison(
		image_results,
		f'Images - {study_name}',
		os.path.join(RESULTS_DIRECTORY, image_save),
	)

	keystroke_results = _run_specs(
		keystroke_specs, keystrokes_loaders, f'Keystroke {study_name}', device
	)
	Trainer.plot_comparison(
		keystroke_results,
		f'Keystrokes - {study_name}',
		os.path.join(RESULTS_DIRECTORY, keystroke_save),
	)


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
		(name, ImageCNN(num_classes=IMAGE_NUM_CLASSES, num_filters=filters), {})
		for name, filters in image_architectures.items()
	]
	keystroke_specs = [
		(name, KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES, num_filters=filters), {})
		for name, filters in keystroke_architectures.items()
	]
	run_study(
		'TYRIMAS 1: Architektūros palyginimas',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
		'img_architecture_comparison.png',
		'ks_architecture_comparison.png',
	)


# Tyrimas 2: Dropout palyginimas
def run_dropout_experiments(image_loaders, keystrokes_loaders, device):
	dropout_values = [0.0, 0.25, 0.5, 0.75]
	image_specs = [
		(f'Dropout={d}', ImageCNN(num_classes=IMAGE_NUM_CLASSES, dropout=d), {})
		for d in dropout_values
	]
	keystroke_specs = [
		(f'Dropout={d}', KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES, dropout=d), {})
		for d in dropout_values
	]
	run_study(
		'TYRIMAS 2: Dropout palyginimas',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
		'img_dropout_comparison.png',
		'ks_dropout_comparison.png',
	)


# Tyrimas 3: Batch Normalization
def run_batch_norm_experiments(image_loaders, keystrokes_loaders, device):
	bn_options = [(False, 'Without BN'), (True, 'With BN')]
	image_specs = [
		(label, ImageCNN(num_classes=IMAGE_NUM_CLASSES, use_batch_norm=use_bn), {})
		for use_bn, label in bn_options
	]
	keystroke_specs = [
		(label, KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES, use_batch_norm=use_bn), {})
		for use_bn, label in bn_options
	]
	run_study(
		'TYRIMAS 3: Batch Normalization',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
		'img_batchnorm_comparison.png',
		'ks_batchnorm_comparison.png',
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
		(str(activation), ImageCNN(num_classes=IMAGE_NUM_CLASSES, activation=activation), {})
		for activation in activations
	]
	keystroke_specs = [
		(
			str(activation),
			KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES, activation=activation),
			{},
		)
		for activation in activations
	]
	run_study(
		'TYRIMAS 4: Aktyvacijos funkcijos',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
		'img_activation_comparison.png',
		'ks_activation_comparison.png',
	)


# Tyrimas 5: Optimizavimo algoritmai
def run_optimizer_experiments(image_loaders, keystrokes_loaders, device):
	image_specs = [
		(str(optimizer), ImageCNN(num_classes=IMAGE_NUM_CLASSES), {'optimizer': optimizer})
		for optimizer in Optimizer
	]
	keystroke_specs = [
		(str(optimizer), KeystrokeCNN(num_classes=KEYSTROKE_NUM_CLASSES), {'optimizer': optimizer})
		for optimizer in Optimizer
	]
	run_study(
		'TYRIMAS 5: Optimizavimo algoritmai',
		image_specs,
		keystroke_specs,
		image_loaders,
		keystrokes_loaders,
		device,
		'img_optimizer_comparison.png',
		'ks_optimizer_comparison.png',
	)


def evaluate_best_model(model, loaders, device, dir_name, display_name, title_name):
	train, val, test, classes = loaders
	print(f'\n--- Geriausias {display_name} modelis ---')

	trainer = Trainer(
		model=model,
		train_loader=train,
		val_loader=val,
		test_loader=test,
		optimizer=Optimizer.ADAM,
		learning_rate=LEARNING_RATE,
		device=device,
	)
	trainer.train(epochs=EPOCHS)
	trainer.evaluate()

	output_dir = os.path.join(RESULTS_DIRECTORY, f'best_{dir_name}_model')
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

	print(f'\n30 pavyzdinių spėjimų ({display_name}):')
	for sample in trainer.get_sample_predictions(num_samples=30, class_names=classes):
		is_correct = '✓' if sample['true'] == sample['predicted'] else '✗'
		print(f'  {is_correct} Tikra: {sample["true"]}, Spėjimas: {sample["predicted"]}')


# Punktas 4: Geriausio modelio įvertinimas
def run_best_model_evaluation(image_loaders, keystrokes_loaders, device):
	print('\n' + '=' * 60)
	print('GERIAUSIO MODELIO ĮVERTINIMAS')
	print('=' * 60)

	best_image_model = ImageCNN(
		num_classes=IMAGE_NUM_CLASSES,
		num_filters=(32, 64, 128),
		activation=ActivationFunction.RELU,
		dropout=0.25,
		use_batch_norm=True,
	)
	evaluate_best_model(best_image_model, image_loaders, device, 'image', 'vaizdų', 'Image')

	best_keystroke_model = KeystrokeCNN(
		num_classes=KEYSTROKE_NUM_CLASSES,
		num_filters=(64, 128),
		activation=ActivationFunction.RELU,
		dropout=0.25,
		use_batch_norm=True,
	)
	evaluate_best_model(
		best_keystroke_model,
		keystrokes_loaders,
		device,
		'keystroke',
		'klavišų paspaudimų',
		'Keystroke',
	)


def main():
	device = get_device()
	os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

	# Užkrauname duomenis
	image_loaders = load_image_data()
	keystrokes_loaders = load_keystroke_data()

	run_architecture_experiments(image_loaders, keystrokes_loaders, device)
	run_dropout_experiments(image_loaders, keystrokes_loaders, device)
	run_batch_norm_experiments(image_loaders, keystrokes_loaders, device)
	run_activation_experiments(image_loaders, keystrokes_loaders, device)
	run_optimizer_experiments(image_loaders, keystrokes_loaders, device)

	run_best_model_evaluation(image_loaders, keystrokes_loaders, device)


if __name__ == '__main__':
	main()
