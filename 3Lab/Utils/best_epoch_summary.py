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
