// ==================== UTILITY FUNCTIONS ====================

// Format currency
function formatCurrency(value, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency
    }).format(value);
}

// Format percentage
function formatPercent(value) {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
}

// Format number with commas
function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

// Show toast notification
function showToast(message, type = 'info') {
    // Implement toast notification
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
