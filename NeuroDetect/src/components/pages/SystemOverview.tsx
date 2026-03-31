import React from 'react';
import { Activity, Lock, ShieldCheck, Server } from 'lucide-react';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../auth/AuthContext';

const SystemOverview: React.FC = () => {
  const { currentTheme, isDarkTheme } = useTheme();
  const { currentUser } = useAuth();

  const border = isDarkTheme ? 'rgba(148, 163, 184, 0.18)' : '#e2e8f0';
  const panel = isDarkTheme ? '#111827' : '#ffffff';
  const panelSoft = isDarkTheme ? '#172033' : '#f8fafc';
  const text = currentTheme.textPrimary;
  const muted = currentTheme.textMuted;

  const cards = [
    { label: 'Access Mode', value: 'Read only', icon: Lock, color: '#f59e0b' },
    { label: 'Current Role', value: currentUser?.role ?? 'viewer', icon: ShieldCheck, color: '#2563eb' },
    { label: 'Scope', value: 'Platform status and system summary', icon: Server, color: '#10b981' },
  ];

  return (
    <div style={{ minHeight: '100vh', background: currentTheme.bgPrimary, color: text, padding: '24px' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ marginBottom: '20px' }}>
          <div style={{ color: muted, fontSize: '.72rem', fontWeight: 700, letterSpacing: '.12em', textTransform: 'uppercase', marginBottom: '8px' }}>
            System
          </div>
          <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 800 }}>System Overview</h1>
          <p style={{ margin: '10px 0 0', color: muted, fontSize: '.95rem' }}>
            Read-only operational overview for admins and analysts.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: '16px', marginBottom: '18px' }}>
          {cards.map((card) => {
            const Icon = card.icon;
            return (
              <div key={card.label} style={{ border: `1px solid ${border}`, borderRadius: '18px', background: panel, padding: '18px' }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '38px', height: '38px', borderRadius: '12px', background: panelSoft, color: card.color, marginBottom: '14px' }}>
                  <Icon size={18} />
                </div>
                <div style={{ color: muted, fontSize: '.72rem', fontWeight: 700, letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: '8px' }}>{card.label}</div>
                <div style={{ fontSize: '1.04rem', fontWeight: 700, textTransform: 'capitalize' }}>{card.value}</div>
              </div>
            );
          })}
        </div>

        <div style={{ border: `1px solid ${border}`, borderRadius: '20px', background: panel, padding: '22px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <Activity size={18} color="#7c3aed" />
            <div style={{ fontSize: '1rem', fontWeight: 700 }}>Read-only note</div>
          </div>
          <p style={{ margin: 0, color: muted, lineHeight: 1.7, fontSize: '.92rem' }}>
            This page is a read-only destination for the new system navigation entry. It is intended to centralize
            platform status, monitoring summaries, and operational context without exposing editing controls.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SystemOverview;