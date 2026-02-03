//! Transaction operations for the Ember backend.

use crate::Ember;
use burn_backend::ops::TransactionOps;

// TransactionOps has default implementations.
impl TransactionOps<Ember> for Ember {}
