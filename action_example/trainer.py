import numpy as np

import updater

class Trainer(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.updater_config = kwargs.pop('updater_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        if not hasattr(updater, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)

        self.update_rule = getattr(updater, self.update_rule)

        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.updater_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.updater_config.items()}
            self.updater_configs[p] = d

    def _step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.updater_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.updater_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = int(N / batch_size)
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(int(num_train / self.batch_size), 1)
        print(iterations_per_epoch)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print('(iteration %d / %d, loss val %f)' % (t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.updater_configs:
                    self.updater_configs[k]['learning_rate'] *= self.lr_decay

            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)

                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(period %d / %d), train_acc :%f, val_acc:%f' % (self.epoch, self.num_epochs, train_acc, val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        self.model.params = self.best_params
